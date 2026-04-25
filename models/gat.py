"""
gat.py  —  Stage 2: Type-Specific Graph Attention Network (Semi-Supervised Anomaly Detection)

Architecture (paper Section IV-D):
  Layer: Type-specific GAT attention (Eq. 12-13)
  Output: Anomaly score per node via sigmoid (Eq. 14)
  Loss:   L_GAT = L_sup + λ * L_unsup  (Eq. 15-17)

Key equations:
  α_ij^r = softmax_k [ LeakyReLU( a_r^T [W_r h_i || W_r h_j] ) ]  (Eq. 12)
  h'_i   = σ( Σ_r Σ_{j∈N_r(i)} α_ij^r * W_r * h_j )              (Eq. 13)
  s_i    = σ( W_f h'_i + b_f )                                      (Eq. 14)
  L_sup  = -Σ_{i∈VL} [ y_i log(s_i) + (1-y_i) log(1-s_i) ]       (Eq. 16)
  L_unsup= Σ_{i,j} A_ij ||s_i - s_j||²                            (Eq. 17)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TypeSpecificGATLayer(nn.Module):
    """
    One GAT layer with per-edge-type transformations (Eq. 12-13).

    For each relation r:
      1. Transform: h_i_r = W_r h_i            (linear, in_dim -> out_dim)
      2. Attention: e_ij_r = LeakyReLU( a_r^T [h_i_r || h_j_r] )
      3. Normalize: α_ij_r = softmax over j ∈ N_r(i)
      4. Aggregate: contrib_r[i] = Σ_j α_ij_r * h_j_r

    Final: h'_i = σ( Σ_r contrib_r[i] ) with shared bias.

    Dense implementation: works on A_r [N, N] matrices.
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
        self.edge_types  = edge_types
        self.out_dim     = out_dim
        self.leaky_slope = leaky_slope

        # W_r: in_dim -> out_dim per edge type
        self.W = nn.ModuleDict({r: nn.Linear(in_dim, out_dim, bias=False) for r in edge_types})

        # a_r: attention vector of size 2*out_dim per edge type (Eq. 12)
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
        H: torch.Tensor,                            # [N, in_dim]
        A_per_type: Dict[str, torch.Tensor],        # {rel -> [N, N] binary adj}
        return_attention: bool = False,
    ):
        """
        Returns H_new [N, out_dim].
        If return_attention=True, also returns {rel -> attention_matrix [N,N]}.

        [Lec 2] Phase 2: When CUDA kernels are available and data is on GPU,
        this method transparently switches from the dense O(N²) PyTorch path
        to our custom CUDA kernels that operate on sparse edge lists (COO),
        using tiled_matmul, gat_attention, and neighbor_aggregation kernels.
        Falls back to the original dense implementation on CPU or when kernels
        are not compiled.
        """
        N = H.shape[0]
        aggregated = torch.zeros(N, self.out_dim, device=H.device, dtype=H.dtype)
        attn_dict = {}

        # [Lec 4] GPGPU: Use custom CUDA kernels during inference (no_grad).
        # During training, we use PyTorch ops so autograd can compute gradients.
        _cuda_kernels = None
        if H.is_cuda and not torch.is_grad_enabled():
            try:
                import gnn_cuda_kernels as _ck
                _cuda_kernels = _ck
            except ImportError:
                pass

        for r in self.edge_types:
            A_r = A_per_type.get(r)             # [N, N]
            if A_r is None:
                continue

            if _cuda_kernels is not None:
                # ═══════════ PHASE 2: CUSTOM CUDA KERNEL PATH ═══════════
                # [Lec 16-17] Step 1: Tiled MatMul for feature projection
                #   Wh = H @ W_r^T   using our shared-memory tiled kernel
                W_t = self.W[r].weight.t().contiguous()     # [in_dim, out_dim]
                Wh  = _cuda_kernels.tiled_matmul(H.contiguous(), W_t)  # [N, out_dim]

                # Extract sparse edge list (COO) from dense adjacency
                row, col = A_r.nonzero(as_tuple=True)
                row_i, col_i = row.long().contiguous(), col.long().contiguous()

                # [Lec 6-7] Step 2: Per-edge GAT attention via thread-per-edge kernel
                #   e_ij = LeakyReLU( a_r^T [Wh_i || Wh_j] )
                #   Note: kernel internally computes attn[:D]*Wh[dst] + attn[D:]*Wh[src],
                #   while PyTorch does attn[:D]*Wh[row] + attn[D:]*Wh[col].
                #   So we pass col as 'row' and row as 'col' to match semantics.
                e_sparse = _cuda_kernels.gat_attention(
                    Wh, col_i, row_i, self.a[r].contiguous(), self.leaky_slope
                )

                # Softmax over neighbors (scatter back to dense for row-wise softmax)
                e_dense = torch.full((N, N), float('-inf'), device=H.device)
                e_dense[row, col] = e_sparse
                alpha = torch.softmax(e_dense, dim=1)
                alpha = torch.nan_to_num(alpha, nan=0.0)

                alpha_sparse = alpha[row, col].contiguous()
                attn_dict[r] = alpha

                # [Lec 9,20] Step 3: Neighbor aggregation with atomicAdd
                #   h'_i = Σ_j α_ij * Wh_j
                #   Kernel: out[dst] += alpha * Wh[src]. We want out[row] += alpha * Wh[col],
                #   so pass col_i as src (row arg) and row_i as dst (col arg).
                agg_r = _cuda_kernels.neighbor_aggregation(
                    Wh, alpha_sparse, col_i, row_i, N
                )
                aggregated = aggregated + agg_r

            else:
                # ═══════════ PHASE 1: DENSE FALLBACK (CPU or no kernels) ═══════════
                Wh = self.W[r](H)                   # [N, out_dim]  → W_r h

                # Build pairwise concatenation efficiently:
                # [Wh_i || Wh_j] for each (i,j) where A_r[i,j]=1
                # In dense mode: broadcast [N, out_dim] -> [N, N, 2*out_dim]
                Wh_i = Wh.unsqueeze(1).expand(N, N, self.out_dim)  # [N, N, out_dim]
                Wh_j = Wh.unsqueeze(0).expand(N, N, self.out_dim)  # [N, N, out_dim]
                pair  = torch.cat([Wh_i, Wh_j], dim=-1)            # [N, N, 2*out_dim]

                # Compute attention logits: e_ij = LeakyReLU(a_r^T [Wh_i || Wh_j])
                e = F.leaky_relu(
                    (pair * self.a[r]).sum(dim=-1),   # [N, N]
                    negative_slope=self.leaky_slope,
                )

                # Mask out non-neighbors:  set to -inf so softmax → 0
                mask = (A_r == 0)
                e = e.masked_fill(mask, float('-inf'))

                # Softmax over neighbors (row-wise) — Eq. 12
                alpha = torch.softmax(e, dim=1)      # [N, N]
                # Fix rows with ALL -inf (isolated nodes): set to 0 instead of NaN
                alpha = torch.nan_to_num(alpha, nan=0.0)

                attn_dict[r] = alpha

                # Weighted aggregation: Σ_j α_ij * (W_r h_j)  — Eq. 13
                aggregated = aggregated + alpha @ Wh  # [N, out_dim]

        if self.bias is not None:
            aggregated = aggregated + self.bias

        if return_attention:
            return aggregated, attn_dict
        return aggregated


class GraphAttentionNetwork(nn.Module):
    """
    Two-layer type-specific GAT for anomaly detection.

    Layer 1: TypeSpecificGATLayer(embed_dim -> hidden_dim)  with ReLU
    Layer 2: TypeSpecificGATLayer(hidden_dim -> hidden_dim) with ReLU
    Output:  Linear(hidden_dim -> 1) + sigmoid  →  anomaly score s_i ∈ [0, 1]
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        edge_types: list,
    ):
        super().__init__()
        self.layer1 = TypeSpecificGATLayer(embed_dim,  hidden_dim, edge_types)
        self.layer2 = TypeSpecificGATLayer(hidden_dim, hidden_dim, edge_types)

        # Final anomaly scoring head: W_f, b_f (Eq. 14)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        Z: torch.Tensor,                        # [N, embed_dim]  — from GAE Stage 1
        A_per_type: Dict[str, torch.Tensor],    # {rel -> [N, N]}
        return_attention: bool = False,
    ):
        """
        Returns anomaly scores s: [N] ∈ (0, 1).
        """
        H1 = F.relu(self.layer1(Z, A_per_type))   # [N, hidden_dim]
        H2, attn = self.layer2(H1, A_per_type, return_attention=True)  # [N, hidden_dim]
        H2 = F.relu(H2)

        # Anomaly score: s_i = σ(W_f h'_i + b_f)  (Eq. 14)
        raw = self.score_head(H2).squeeze(-1)      # [N]
        s   = torch.sigmoid(raw)                   # [N]   ∈ (0,1)

        if return_attention:
            return s, attn
        return s

    @staticmethod
    def supervised_loss(
        s: torch.Tensor,           # [N] anomaly scores
        y: torch.Tensor,           # [N] ground truth labels
        labeled_mask: torch.Tensor,# [N] bool
    ) -> torch.Tensor:
        """
        L_sup = -Σ_{i∈VL} [y_i log(s_i) + (1-y_i) log(1-s_i)]  (Eq. 16)
        Handles class imbalance via pos_weight.
        """
        s_l = s[labeled_mask]
        y_l = y[labeled_mask].float()

        if s_l.numel() == 0:
            return torch.tensor(0.0, device=s.device)

        # Reweight positives (anomalies are rare)
        n_pos = y_l.sum().clamp(min=1)
        n_neg = y_l.numel() - n_pos
        pos_weight = torch.as_tensor(n_neg / n_pos, device=s.device).clamp(max=10.0)

        return F.binary_cross_entropy(
            s_l, y_l,
            weight=torch.where(y_l == 1, pos_weight, torch.ones_like(y_l)),
            reduction='mean',
        )

    @staticmethod
    def unsupervised_loss(
        s: torch.Tensor,        # [N] anomaly scores
        A: torch.Tensor,        # [N, N] full adjacency
    ) -> torch.Tensor:
        """
        L_unsup = Σ_{i,j} A_ij ||s_i - s_j||²  (Eq. 17)
        Encourages connected nodes to have similar anomaly scores.

        [Lec 14] Phase 2: Uses custom CUDA kernel with warp-level shuffle
        reduction (__shfl_down_sync) + atomicAdd for parallel summation
        over all edges.
        """
        if s.is_cuda and not torch.is_grad_enabled():
            try:
                import gnn_cuda_kernels
                # Extract sparse edge list from dense adjacency
                row, col = A.nonzero(as_tuple=True)
                # [Lec 14] Custom kernel: parallel reduction over edges
                sum_loss = gnn_cuda_kernels.smoothness_loss(
                    s.contiguous(), row.long().contiguous(), col.long().contiguous()
                )
                # Normalize to match the dense mean reduction (sum / N²)
                return sum_loss / A.numel()
            except ImportError:
                pass

        # ═══════════ PHASE 1: DENSE FALLBACK ═══════════
        # s_i - s_j for all pairs: [N, N]
        diff = s.unsqueeze(1) - s.unsqueeze(0)    # [N, N]
        sq   = diff ** 2                           # [N, N]
        return (A * sq).mean()  # consistent with L_sup which also uses mean

    @staticmethod
    def combined_loss(
        s: torch.Tensor,
        y: torch.Tensor,
        labeled_mask: torch.Tensor,
        A: torch.Tensor,
        lam: float = 0.5,
    ) -> torch.Tensor:
        """
        L_GAT = L_sup + λ * L_unsup  (Eq. 15)
        """
        l_sup   = GraphAttentionNetwork.supervised_loss(s, y, labeled_mask)
        l_unsup = GraphAttentionNetwork.unsupervised_loss(s, A)
        return l_sup + lam * l_unsup, l_sup, l_unsup
