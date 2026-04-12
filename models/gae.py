"""
gae.py  —  Stage 1: Graph AutoEncoder (Unsupervised Representation Learning)

Architecture (paper Section IV-C):
  Encoder: 2-layer GCN with type-specific weight matrices (Eq. 7)
  Decoder: Inner-product  Â = σ(Z Zᵀ)  (Eq. 8)  or asymmetric MLP (Section 2.6)
  Loss:    Binary cross-entropy over all node pairs / sampled edge pairs (Eq. 9)

Key equations:
  h_v^(l) = σ( Σ_r Σ_{u ∈ N_r(v)} [ 1/√(|N(v)||N(u)|) * W_r^(l) * h_u^(l-1) ] + b^(l) )
  Â       = σ( Z Zᵀ )                          — inner-product decoder
  Â_ij    = σ( MLP([z_i || z_j]) )             — asymmetric MLP decoder
  L_GAE   = -Σ_{i,j} [ A_ij log(Â_ij) + (1-A_ij) log(1-Â_ij) ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from models.mlp_decoder import MLPDecoder


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
    ) -> torch.Tensor:
        """
        Returns H_new [N, out_dim].
        """
        # We output out_dim, so init shape is (N, out_dim)
        out_dim = list(self.W.values())[0].weight.shape[0]
        aggregated = torch.zeros(H.shape[0], out_dim, device=H.device, dtype=H.dtype)

        for r in self.edge_types:
            A_r = A_norm_per_type.get(r)
            if A_r is None:
                continue
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
    ) -> torch.Tensor:
        """
        Returns Z [N, embed_dim].
        """
        H1 = F.relu(self.layer1(H, A_norm_per_type))   # Eq. 7 layer 1
        Z  = self.layer2(H1, A_norm_per_type)           # Eq. 7 layer 2 (no activation)
        return Z


class GAEDecoder(nn.Module):
    """
    Inner-product decoder: Â = σ(Z Zᵀ)  (Eq. 8).
    """

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: [N, embed_dim]
        Returns raw logits: [N, N]  (reconstructed adjacency)
        """
        return Z @ Z.t()


class GraphAutoEncoder(nn.Module):
    """
    Full GAE = Encoder + Decoder.

    Supports two decoder types:
      'inner_product' (default) — symmetric Â = σ(Z Zᵀ), computes over all N^2 pairs.
      'mlp'                     — asymmetric MLP on sampled directed edge pairs,
                                  suitable for directed heterogeneous graphs.

    Usage:
        gae = GraphAutoEncoder(feat_dim=19, hidden_dim=64, embed_dim=128,
                               edge_types=[...], decoder_type='mlp')
        Z, loss = gae.compute_reconstruction_loss(H, A, A_norm_per_type)
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        embed_dim: int,
        edge_types: list,
        decoder_type: str = 'inner_product',   # 'inner_product' | 'mlp'
    ):
        super().__init__()
        self.decoder_type = decoder_type
        self.encoder      = GAEEncoder(feat_dim, hidden_dim, embed_dim, edge_types)
        if decoder_type == 'mlp':
            self.decoder = MLPDecoder(embed_dim, hidden_dim)
        else:
            self.decoder = GAEDecoder()

    def encode(
        self,
        H: torch.Tensor,
        A_norm_per_type: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.encoder(H, A_norm_per_type)

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

    def compute_reconstruction_loss(
        self,
        H: torch.Tensor,
        A: torch.Tensor,
        A_norm_per_type: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified encode -> decode -> loss step for both decoder types.

        Returns:
            Z    — frozen embeddings [N, embed_dim]
            loss — reconstruction loss scalar
        """
        Z = self.encode(H, A_norm_per_type)
        if self.decoder_type == 'mlp':
            edge_index, labels = MLPDecoder.sample_edges(A)
            logits = self.decoder(Z, edge_index)
            loss   = MLPDecoder.reconstruction_loss(logits, labels)
        else:
            A_hat_logits = self.decode(Z)
            loss = GraphAutoEncoder.reconstruction_loss(A, A_hat_logits)
        return Z, loss

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
