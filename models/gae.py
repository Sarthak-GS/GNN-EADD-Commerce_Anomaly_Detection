"""
gae.py  —  Stage 1: Graph AutoEncoder (Unsupervised Representation Learning)

Architecture (paper Section IV-C):
  Encoder: 2-layer GCN with type-specific weight matrices (Eq. 7)
  Decoder: Inner-product decoder  Â = σ(Z Zᵀ)  (Eq. 8)
  Loss:    Binary cross-entropy over all node pairs (Eq. 9)

Key equations:
  h_v^(l) = σ( Σ_r Σ_{u ∈ N_r(v)} [ 1/√(|N(v)||N(u)|) * W_r^(l) * h_u^(l-1) ] + b^(l) )
  Â       = σ( Z Zᵀ )
  L_GAE   = -Σ_{i,j} [ A_ij log(Â_ij) + (1-A_ij) log(1-Â_ij) ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class TypeSpecificGCNLayer(nn.Module):
    """
    One GCN layer with type-specific weight matrices (Eq. 7).

    For each edge type r in R:
        contribution_r[v] = Σ_{u ∈ N_r(v)} [ 1/√(|N(v)||N(u)|) * W_r * h_u ]

    Then sum across all r and add bias:
        h_v^(l) = σ( Σ_r contribution_r[v] + b )

    This runs in dense matrix mode (A_norm is [N, N]) for clarity and
    correctness on small-medium graphs (≤ 10k nodes).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_types: list,       # list of relation names, e.g. ['purchase','sell','interact']
        bias: bool = True,
    ):
        super().__init__()
        self.edge_types = edge_types
        # One weight matrix per edge type  W_r ∈ R^{in_dim × out_dim}
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
        H: torch.Tensor,               # [N, in_dim]  current node features
        A_norm_per_type: Dict[str, torch.Tensor],  # {rel -> [N, N] normalized adj}
        coo_cache: Dict[str, tuple] = None,  # pre-computed {rel -> (row, col, alpha)}
    ) -> torch.Tensor:
        """
        Returns H_new [N, out_dim].

        [Lec 2] Phase 2: When CUDA kernels are available and data is on GPU,
        uses custom neighbor_aggregation kernel for message passing and
        tiled_matmul kernel for the W_r projection. Falls back to dense
        matmul on CPU or when kernels are not compiled.

        coo_cache: Optional pre-computed COO edges. Pass this during repeated
        inference to skip the expensive A_r.nonzero() extraction every call.
        """
        # We output out_dim, so init shape is (N, out_dim)
        out_dim = list(self.W.values())[0].weight.shape[0]
        N = H.shape[0]
        aggregated = torch.zeros(N, out_dim, device=H.device, dtype=H.dtype)

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
            A_r = A_norm_per_type.get(r)
            if A_r is None:
                continue

            if _cuda_kernels is not None:
                # ═══════════ PHASE 2: CUSTOM CUDA KERNEL PATH ═══════════
                # Use pre-computed COO cache if available (avoids O(N²) nonzero scan)
                if coo_cache and r in coo_cache:
                    row_i, col_i, alpha_sparse = coo_cache[r]
                else:
                    row, col = A_r.nonzero(as_tuple=True)
                    row_i, col_i = row.long().contiguous(), col.long().contiguous()
                    alpha_sparse = A_r[row, col].contiguous()

                # [Lec 9,20] Step 1: Neighbor Aggregation (Message Passing)
                #   msg_v = Σ_{u ∈ N_r(v)} α_vu * h_u
                #   Kernel: out[dst] += alpha * H[src]. We want out[row] += alpha * H[col],
                #   so pass col_i as src (row arg) and row_i as dst (col arg).
                msg = _cuda_kernels.neighbor_aggregation(
                    H.contiguous(), alpha_sparse, col_i, row_i, N
                )  # [N, in_dim]

                # [Lec 16-17] Step 2: Feature projection W_r
                #   out_v = msg_v @ W_r^T
                #   Use cuBLAS (PyTorch @) for small projections (faster than tiled kernel)
                #   Our tiled kernel shines on large square matrices like Z@Z^T in decoder
                aggregated = aggregated + self.W[r](msg)    # [N, out_dim]

            else:
                # ═══════════ PHASE 1: DENSE FALLBACK ═══════════
                # Message: A_norm_r [N, N] @ H [N, in_dim] → [N, in_dim]  then W_r
                msg = A_r @ H                  # [N, in_dim]
                aggregated = aggregated + self.W[r](msg)    # [N, out_dim]

        if self.bias is not None:
            aggregated = aggregated + self.bias

        return aggregated


class GAEEncoder(nn.Module):
    """
    Two-layer GCN encoder (paper Section V-4: embedding_dim=128).

    Input:  H^(0) — concatenated all-node feature matrix [N, feat_dim]
    Output: Z     — latent embeddings [N, embed_dim]
    """

    def __init__(self, feat_dim: int, hidden_dim: int, embed_dim: int, edge_types: list):
        super().__init__()
        self.layer1 = TypeSpecificGCNLayer(feat_dim,   hidden_dim, edge_types)
        self.layer2 = TypeSpecificGCNLayer(hidden_dim, embed_dim,  edge_types)

    def forward(
        self,
        H: torch.Tensor,                            # [N, feat_dim]
        A_norm_per_type: Dict[str, torch.Tensor],   # {rel -> [N, N]}
        coo_cache: Dict[str, tuple] = None,
    ) -> torch.Tensor:
        """
        Returns Z [N, embed_dim].
        """
        H1 = F.relu(self.layer1(H, A_norm_per_type, coo_cache))   # Eq. 7 layer 1
        Z  = self.layer2(H1, A_norm_per_type, coo_cache)           # Eq. 7 layer 2 (no activation)
        return Z


class GAEDecoder(nn.Module):
    """
    Inner-product decoder: Â = σ(Z Zᵀ)  (Eq. 8).

    [Lec 16-17] Phase 2: Uses custom tiled_matmul CUDA kernel for the
    Z @ Zᵀ computation, leveraging shared memory tiling.
    """

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: [N, embed_dim]
        Returns raw logits: [N, N]  (reconstructed adjacency)
        """
        if Z.is_cuda and not torch.is_grad_enabled():
            try:
                import gnn_cuda_kernels
                # [Lec 16-17] Tiled MatMul: Z @ Z^T via shared memory tiles
                return gnn_cuda_kernels.tiled_matmul(
                    Z.contiguous(), Z.t().contiguous()
                )
            except ImportError:
                pass
        return Z @ Z.t()


class GraphAutoEncoder(nn.Module):
    """
    Full GAE = Encoder + Decoder.

    Usage:
        gae = GraphAutoEncoder(feat_dim=19, hidden_dim=64, embed_dim=128, edge_types=[...])
        Z      = gae.encode(H, A_norm_per_type)
        A_hat  = gae.decode(Z)
        loss   = gae.reconstruction_loss(A, A_hat)
    """

    def __init__(self, feat_dim: int, hidden_dim: int, embed_dim: int, edge_types: list):
        super().__init__()
        self.encoder = GAEEncoder(feat_dim, hidden_dim, embed_dim, edge_types)
        self.decoder = GAEDecoder()

    def encode(
        self,
        H: torch.Tensor,
        A_norm_per_type: Dict[str, torch.Tensor],
        coo_cache: Dict[str, tuple] = None,
    ) -> torch.Tensor:
        return self.encoder(H, A_norm_per_type, coo_cache)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        return self.decoder(Z)

    def forward(
        self,
        H: torch.Tensor,
        A_norm_per_type: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (Z, A_hat_logits)."""
        Z            = self.encode(H, A_norm_per_type)
        A_hat_logits = self.decode(Z)
        return Z, A_hat_logits

    @staticmethod
    def reconstruction_loss(A: torch.Tensor, A_hat_logits: torch.Tensor) -> torch.Tensor:
        """
        Binary cross-entropy reconstruction loss (Eq. 9):
            L_GAE = -Σ_{i,j} [ A_ij log(Â_ij) + (1-A_ij) log(1-Â_ij) ]

        We use torch's BCEWithLogits which is numerically stable.
        pos_weight accounts for the class imbalance (most pairs have A_ij=0).
        """
        # Count positives for reweighting
        n_pos = A.sum()
        n_neg = A.numel() - n_pos
        pos_weight = (n_neg / (n_pos + 1e-8)).clamp(max=50.0)

        loss = F.binary_cross_entropy_with_logits(
            input  = A_hat_logits,
            target = A,
            pos_weight = torch.as_tensor(float(pos_weight), device=A.device),
            reduction  = 'mean',
        )
        return loss
