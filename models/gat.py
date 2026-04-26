import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TypeSpecificGATLayer(nn.Module):
    """
    Graph Attention Layer with per-edge-type weights.
    Paying attention is hard, especially to graphs :)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_types: list,
        leaky_slope: float = 0.2,
        bias: bool = True,
    ):
        super().__init__()
        self.edge_types = edge_types
        self.out_dim = out_dim
        self.leaky_slope = leaky_slope

        self.W = nn.ModuleDict({r: nn.Linear(in_dim, out_dim, bias=False) for r in edge_types})
        self.a = nn.ParameterDict({
            r: nn.Parameter(torch.empty(2 * out_dim)) for r in edge_types
        })
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        for lin in self.W.values():
            nn.init.xavier_uniform_(lin.weight)
        for a in self.a.values():
            nn.init.xavier_uniform_(a.view(1, -1))

    def forward(
        self,
        H: torch.Tensor,
        A_per_type: Dict[str, torch.Tensor],
        return_attention: bool = False,
        coo_cache: Dict[str, tuple] = None,
    ):
        N = H.shape[0]
        aggregated = torch.zeros(N, self.out_dim, device=H.device, dtype=H.dtype)
        attn_dict = {}

        _cuda_kernels = None
        if H.is_cuda and not torch.is_grad_enabled():
            try:
                import gnn_cuda_kernels as _ck
                _cuda_kernels = _ck
            except ImportError:
                pass

        for r in self.edge_types:
            A_r = A_per_type.get(r)
            if A_r is None:
                continue

            if _cuda_kernels is not None:
                W_t = self.W[r].weight.t().contiguous()
                Wh = _cuda_kernels.tiled_matmul(H.contiguous(), W_t)

                if coo_cache and r in coo_cache:
                    row, col = coo_cache[r][:2]
                else:
                    row, col = A_r.nonzero(as_tuple=True)
                
                row_i, col_i = row.long().contiguous(), col.long().contiguous()
                e_sparse = _cuda_kernels.gat_attention(
                    Wh, col_i, row_i, self.a[r].contiguous(), self.leaky_slope
                )

                row_dst = row
                e_max = torch.full((N,), float('-inf'), device=H.device)
                e_max.scatter_reduce_(0, row_dst, e_sparse, reduce='amax', include_self=True)
                e_shifted = (e_sparse - e_max[row_dst]).exp()
                e_sum = torch.zeros(N, device=H.device)
                e_sum.scatter_add_(0, row_dst, e_shifted)
                alpha_sparse = (e_shifted / e_sum[row_dst].clamp(min=1e-12)).contiguous()
                attn_dict[r] = None

                agg_r = _cuda_kernels.neighbor_aggregation(Wh, alpha_sparse, col_i, row_i, N)
                aggregated = aggregated + agg_r
            else:
                Wh = self.W[r](H)
                if coo_cache and r in coo_cache:
                    row, col = coo_cache[r][:2]
                else:
                    row, col = A_r.nonzero(as_tuple=True)

                if row.numel() > 0:
                    pair = torch.cat([Wh[row], Wh[col]], dim=-1)
                    e_sparse = F.leaky_relu((pair * self.a[r]).sum(dim=-1), self.leaky_slope)

                    e_max = torch.full((N,), float('-inf'), device=H.device, dtype=H.dtype)
                    e_max.scatter_reduce_(0, row, e_sparse, reduce='amax', include_self=True)
                    e_shifted = (e_sparse - e_max[row]).exp()
                    e_sum = torch.zeros(N, device=H.device, dtype=H.dtype)
                    e_sum.scatter_add_(0, row, e_shifted)
                    alpha_sparse = e_shifted / e_sum[row].clamp(min=1e-12)

                    agg_r = torch.zeros(N, self.out_dim, device=H.device, dtype=H.dtype)
                    agg_r.scatter_add_(0, row.unsqueeze(1).expand(-1, self.out_dim), alpha_sparse.unsqueeze(1) * Wh[col])
                    attn_dict[r] = None
                else:
                    agg_r = torch.zeros(N, self.out_dim, device=H.device, dtype=H.dtype)
                    attn_dict[r] = None
                aggregated = aggregated + agg_r

        if self.bias is not None:
            aggregated = aggregated + self.bias
        return (aggregated, attn_dict) if return_attention else aggregated


class GraphAttentionNetwork(nn.Module):
    """
    Two-layer GAT for suspicious behavior detection.
    Spying on nodes like a nosy neighbor :)
    """

    def __init__(self, embed_dim: int, hidden_dim: int, edge_types: list):
        super().__init__()
        self.layer1 = TypeSpecificGATLayer(embed_dim, hidden_dim, edge_types)
        self.layer2 = TypeSpecificGATLayer(hidden_dim, hidden_dim, edge_types)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, Z, A_per_type, return_attention=False, coo_cache=None):
        H1 = F.relu(self.layer1(Z, A_per_type, coo_cache=coo_cache))
        H2, attn = self.layer2(H1, A_per_type, return_attention=True, coo_cache=coo_cache)
        H2 = F.relu(H2)
        s = torch.sigmoid(self.score_head(H2).squeeze(-1))
        return (s, attn) if return_attention else s

    @staticmethod
    def supervised_loss(s, y, mask):
        s_l, y_l = s[mask], y[mask].float()
        if s_l.numel() == 0: return torch.tensor(0.0, device=s.device)
        n_pos = y_l.sum().clamp(min=1)
        pos_weight = ((y_l.numel() - n_pos) / n_pos).clamp(max=10.0)
        return F.binary_cross_entropy(s_l, y_l, weight=torch.where(y_l == 1, pos_weight, torch.ones_like(y_l)), reduction='mean')

    @staticmethod
    def unsupervised_loss(s, A):
        if s.is_cuda and not torch.is_grad_enabled():
            try:
                import gnn_cuda_kernels as _ck
                row, col = A.nonzero(as_tuple=True)
                return _ck.smoothness_loss(s.contiguous(), row.long().contiguous(), col.long().contiguous()) / A.numel()
            except ImportError:
                pass
        row, col = A.nonzero(as_tuple=True)
        if A.numel() == 0: return torch.tensor(0.0, device=s.device)
        return ((s[row] - s[col]) ** 2).sum() / A.numel()

    @staticmethod
    def combined_loss(s, y, mask, A, lam=0.5):
        l_sup = GraphAttentionNetwork.supervised_loss(s, y, mask)
        l_unsup = GraphAttentionNetwork.unsupervised_loss(s, A)
        return l_sup + lam * l_unsup, l_sup, l_unsup
