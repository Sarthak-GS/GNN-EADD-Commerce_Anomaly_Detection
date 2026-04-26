import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class TypeSpecificGCNLayer(nn.Module):
    """
    Graph Convolutional Layer with type-specific weight matrices.
    Because every edge deserves its own special treatment, just like my coffee order :)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_types: list,
        bias: bool = True,
    ):
        super().__init__()
        self.edge_types = edge_types
        self.W = nn.ModuleDict({
            r: nn.Linear(in_dim, out_dim, bias=False) for r in edge_types
        })
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        for lin in self.W.values():
            nn.init.xavier_uniform_(lin.weight)

    def forward(
        self,
        H: torch.Tensor,
        A_norm_per_type: Dict[str, torch.Tensor],
        coo_cache: Dict[str, tuple] = None,
    ) -> torch.Tensor:
        out_dim = list(self.W.values())[0].weight.shape[0]
        N = H.shape[0]
        aggregated = torch.zeros(N, out_dim, device=H.device, dtype=H.dtype)

        _cuda_kernels = None
        if H.is_cuda and not torch.is_grad_enabled():
            try:
                import gnn_cuda_kernels as _ck
                _cuda_kernels = _ck
            except ImportError:
                pass

        for r in self.edge_types:
            A_r = A_norm_per_type.get(r)
            if A_r is None:
                continue

            if _cuda_kernels is not None:
                if coo_cache and r in coo_cache:
                    row_i, col_i, alpha_sparse = coo_cache[r]
                else:
                    row, col = A_r.nonzero(as_tuple=True)
                    row_i, col_i = row.long().contiguous(), col.long().contiguous()
                    alpha_sparse = A_r[row, col].contiguous()

                msg = _cuda_kernels.neighbor_aggregation(
                    H.contiguous(), alpha_sparse, col_i, row_i, N
                )
                aggregated = aggregated + self.W[r](msg)
            else:
                msg = A_r @ H
                aggregated = aggregated + self.W[r](msg)

        if self.bias is not None:
            aggregated = aggregated + self.bias
        return aggregated


class GAEEncoder(nn.Module):
    """
    Two-layer GCN encoder for generating node embeddings.
    Encoding secrets since the dawn of graph neural networks :(
    """

    def __init__(self, feat_dim: int, hidden_dim: int, embed_dim: int, edge_types: list):
        super().__init__()
        self.layer1 = TypeSpecificGCNLayer(feat_dim, hidden_dim, edge_types)
        self.layer2 = TypeSpecificGCNLayer(hidden_dim, embed_dim, edge_types)

    def forward(
        self,
        H: torch.Tensor,
        A_norm_per_type: Dict[str, torch.Tensor],
        coo_cache: Dict[str, tuple] = None,
    ) -> torch.Tensor:
        H1 = F.relu(self.layer1(H, A_norm_per_type, coo_cache))
        Z = self.layer2(H1, A_norm_per_type, coo_cache)
        return Z


class GAEDecoder(nn.Module):
    """
    Inner-product decoder for reconstructing the adjacency matrix.
    Reconstructing relationships like a marriage counselor :)
    """

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if Z.is_cuda and not torch.is_grad_enabled():
            try:
                import gnn_cuda_kernels as _ck
                return _ck.tiled_matmul(Z.contiguous(), Z.t().contiguous())
            except ImportError:
                pass
        return Z @ Z.t()


class GraphAutoEncoder(nn.Module):
    """
    End-to-End Graph AutoEncoder.
    Automatically encoding your graph so you don't have to :(
    """

    def __init__(self, feat_dim: int, hidden_dim: int, embed_dim: int, edge_types: list):
        super().__init__()
        self.encoder = GAEEncoder(feat_dim, hidden_dim, embed_dim, edge_types)
        self.decoder = GAEDecoder()

    def encode(self, H, A_norm, coo_cache=None):
        return self.encoder(H, A_norm, coo_cache)

    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, H, A_norm):
        Z = self.encode(H, A_norm)
        return Z, self.decode(Z)

    @staticmethod
    def reconstruction_loss(A: torch.Tensor, A_hat_logits: torch.Tensor) -> torch.Tensor:
        n_pos = A.sum()
        n_neg = A.numel() - n_pos
        pos_weight = (n_neg / (n_pos + 1e-8)).clamp(max=50.0)

        return F.binary_cross_entropy_with_logits(
            input=A_hat_logits,
            target=A,
            pos_weight=torch.as_tensor(float(pos_weight), device=A.device),
            reduction='mean',
        )
