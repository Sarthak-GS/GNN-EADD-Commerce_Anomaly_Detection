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
        coo_cache: Dict[str, tuple] = None,         # pre-computed {rel -> (row, col)}
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

                # Use pre-computed COO cache if available (avoids O(N²) nonzero scan)
                if coo_cache and r in coo_cache:
                    row_i, col_i = coo_cache[r][:2]
                    row, col = row_i, col_i
                else:
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

                # ── Sparse segment-wise softmax (O(E), no N×N matrix!) ──────
                # Group logits by destination node (row), subtract max for
                # numerical stability, then normalise — standard trick.
                row_dst = row                           # destination nodes [E]
                # max per destination node
                e_max = torch.full((N,), float('-inf'), device=H.device)
                e_max.scatter_reduce_(0, row_dst, e_sparse, reduce='amax', include_self=True)
                e_shifted = (e_sparse - e_max[row_dst]).exp()   # [E]
                # sum per destination node
                e_sum = torch.zeros(N, device=H.device)
                e_sum.scatter_add_(0, row_dst, e_shifted)
                alpha_sparse = (e_shifted / e_sum[row_dst].clamp(min=1e-12)).contiguous()
                attn_dict[r] = None   # not building dense attn in CUDA path (saves memory)

                # [Lec 9,20] Step 3: Neighbor aggregation with atomicAdd
                #   h'_i = Σ_j α_ij * Wh_j
                #   Kernel: out[dst] += alpha * Wh[src]. We want out[row] += alpha * Wh[col],
                #   so pass col_i as src (row arg) and row_i as dst (col arg).
                agg_r = _cuda_kernels.neighbor_aggregation(
                    Wh, alpha_sparse, col_i, row_i, N
                )
                aggregated = aggregated + agg_r

            else:
                # ═══════════ PHASE 1: SEMI-SPARSE FALLBACK (CPU / training) ═══════════
                Wh = self.W[r](H)                   # [N, out_dim]

                if coo_cache and r in coo_cache:
                    row, col = coo_cache[r][:2]
                else:
                    row, col = A_r.nonzero(as_tuple=True)
                if row.numel() > 0:
                    # Compute attention only on existing edges — O(E), not O(N²)
                    Wh_row = Wh[row]                              # [E, out_dim]
                    Wh_col = Wh[col]                              # [E, out_dim]
                    pair   = torch.cat([Wh_row, Wh_col], dim=-1) # [E, 2*out_dim]

                    # e_ij = LeakyReLU(a_r^T [Wh_i || Wh_j])  (Eq. 12)
                    e_sparse = F.leaky_relu(
                        (pair * self.a[r]).sum(dim=-1),
                        negative_slope=self.leaky_slope
                    )  # [E]

                    # Sparse segment-wise softmax — no N×N matrix ever allocated
                    e_max = torch.full((N,), float('-inf'), device=H.device, dtype=H.dtype)
                    e_max.scatter_reduce_(0, row, e_sparse, reduce='amax', include_self=True)
                    e_shifted = (e_sparse - e_max[row]).exp()
                    e_sum = torch.zeros(N, device=H.device, dtype=H.dtype)
                    e_sum.scatter_add_(0, row, e_shifted)
                    alpha_sparse = e_shifted / e_sum[row].clamp(min=1e-12)  # [E]

                    # Weighted scatter-add aggregation: out[row] += alpha * Wh[col]
                    agg_r = torch.zeros(N, self.out_dim, device=H.device, dtype=H.dtype)
                    agg_r.scatter_add_(
                        0,
                        row.unsqueeze(1).expand(-1, self.out_dim),
                        alpha_sparse.unsqueeze(1) * Wh_col
                    )

                    attn_dict[r] = None   # sparse path: no dense attn matrix stored
                else:
                    agg_r = torch.zeros(N, self.out_dim, device=H.device, dtype=H.dtype)
                    attn_dict[r] = None

                aggregated = aggregated + agg_r

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
        coo_cache: Dict[str, tuple] = None,
    ):
        """
        Returns anomaly scores s: [N] ∈ (0, 1).
        """
        H1 = F.relu(self.layer1(Z, A_per_type, coo_cache=coo_cache))   # [N, hidden_dim]
        H2, attn = self.layer2(H1, A_per_type, return_attention=True, coo_cache=coo_cache)  # [N, hidden_dim]
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

        # ═══════════ PHASE 1: SEMI-DENSE FALLBACK ═══════════
        # Calculate loss only on existing edges to avoid O(N^2) memory
        row, col = A.nonzero(as_tuple=True)
        diff = s[row] - s[col]
        sq = diff ** 2
        
        if A.numel() == 0:
            return torch.tensor(0.0, device=s.device)
        return sq.sum() / A.numel()  # consistent with mean over [N, N] matrix

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
