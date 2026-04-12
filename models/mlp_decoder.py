"""
mlp_decoder.py  —  Asymmetric MLP decoder for the GAE (Section 2.6 of paper).

Replaces the symmetric inner-product decoder  Ahat = sigma(Z Z^T)  with a
two-layer MLP operating on concatenated node embedding pairs:

    MLP([z_i || z_j])  ->  scalar logit for edge i -> j

Unlike Z Z^T, this is NOT symmetric: edge (i->j) != edge (j->i),
which is appropriate for a directed heterogeneous e-commerce graph.

Training samples positive edges from the adjacency and random negative edges
to avoid the O(N^2) cost of evaluating all pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MLPDecoder(nn.Module):
    """
    Two-layer asymmetric MLP decoder for directed edge reconstruction.

    Input:   concatenated pair [z_i || z_j]  ->  [2 * embed_dim]
    Output:  scalar logit for directed edge i -> j

    Because the MLP sees (z_i, z_j) as an ordered pair, it can learn
    that u->v and v->u have different anomaly implications.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        Z: torch.Tensor,           # [N, embed_dim]
        edge_index: torch.Tensor,  # [2, E]  (src row, dst row)
    ) -> torch.Tensor:
        """
        Returns raw logits: [E]  — one per queried directed edge pair.
        """
        src, dst = edge_index[0], edge_index[1]
        pair = torch.cat([Z[src], Z[dst]], dim=-1)   # [E, 2*embed_dim]
        return self.net(pair).squeeze(-1)             # [E]

    @staticmethod
    def sample_edges(
        A: torch.Tensor,           # [N, N] dense adjacency
        neg_ratio: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples training edges: all positives + neg_ratio * E_pos negatives.

        Returns:
            edge_index: [2, E_pos + E_neg]
            labels:     [E_pos + E_neg]  float (1.0 for pos, 0.0 for neg)
        """
        pos      = A.nonzero(as_tuple=False)      # [E_pos, 2]
        E_pos    = pos.shape[0]
        N        = A.shape[0]
        n_neg    = E_pos * neg_ratio

        neg_i    = torch.randint(0, N, (n_neg,), device=A.device)
        neg_j    = torch.randint(0, N, (n_neg,), device=A.device)
        neg      = torch.stack([neg_i, neg_j], dim=1)   # [E_neg, 2]

        edges  = torch.cat([pos, neg], dim=0).t().contiguous()   # [2, E]
        labels = torch.cat([
            torch.ones(E_pos,  device=A.device),
            torch.zeros(n_neg, device=A.device),
        ])
        return edges, labels

    @staticmethod
    def reconstruction_loss(
        logits: torch.Tensor,   # [E]  raw MLP outputs
        labels: torch.Tensor,   # [E]  0.0 or 1.0
    ) -> torch.Tensor:
        """
        Weighted BCE loss over sampled edge pairs.
        pos_weight corrects for the negative oversampling.
        """
        n_pos      = labels.sum().clamp(min=1)
        n_neg      = labels.numel() - n_pos
        pos_weight = (n_neg / n_pos).clamp(max=50.0)
        return F.binary_cross_entropy_with_logits(
            logits, labels,
            pos_weight=torch.as_tensor(float(pos_weight), device=logits.device),
            reduction='mean',
        )
