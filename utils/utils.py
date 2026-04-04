"""
utils.py  —  Shared utilities:
  - build_per_type_adj_matrices   (constructs dense A_r for each relation)
  - build_homogeneous_features    (concatenates node features into H^(0))
  - compute_metrics               (AUC-ROC, AUC-PR, F1, Precision@K, Recall@K)
  - set_seed
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)
from typing import Dict, Tuple
from data.graph_builder import HeteroGraph


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_per_type_adj_matrices(
    graph: HeteroGraph,
    normalized: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Builds one N×N adjacency matrix per edge relation type for the GAT.

    Relations mapped to strings used as keys in W / a dicts:
      'purchase'  : product ↔ user
      'sell'      : seller  ↔ product
      'interact'  : user    ↔ user
      'self'      : self-loop (all nodes)

    Returns:
        {relation_name -> [N, N] (optionally D-normalized)}
    """
    N_P = graph.num_nodes_per_type['product']
    N_U = graph.num_nodes_per_type['user']
    N_S = graph.num_nodes_per_type['seller']
    N   = N_P + N_U + N_S

    A_dict = {r: torch.zeros(N, N) for r in ['purchase', 'sell', 'interact', 'self']}

    # Self-loops
    A_dict['self'].fill_diagonal_(1.0)

    # Purchase: P ↔ U
    ei = graph.edge_index_dict.get(('product', 'purchase', 'user'), torch.zeros(2, 0, dtype=torch.long))
    for p, u in zip(ei[0].tolist(), ei[1].tolist()):
        gp, gu = p, N_P + u
        A_dict['purchase'][gp, gu] = 1.0
        A_dict['purchase'][gu, gp] = 1.0

    # Sell: S ↔ P
    ei = graph.edge_index_dict.get(('seller', 'sell', 'product'), torch.zeros(2, 0, dtype=torch.long))
    for s, p in zip(ei[0].tolist(), ei[1].tolist()):
        gs, gp = N_P + N_U + s, p
        A_dict['sell'][gs, gp] = 1.0
        A_dict['sell'][gp, gs] = 1.0

    # Interact: U ↔ U
    ei = graph.edge_index_dict.get(('user', 'interact', 'user'), torch.zeros(2, 0, dtype=torch.long))
    for u1, u2 in zip(ei[0].tolist(), ei[1].tolist()):
        gu1, gu2 = N_P + u1, N_P + u2
        A_dict['interact'][gu1, gu2] = 1.0

    if normalized:
        return {r: _sym_normalize(A) for r, A in A_dict.items()}
    return A_dict


def _sym_normalize(A: torch.Tensor) -> torch.Tensor:
    """D^{-1/2} A D^{-1/2} symmetric normalization (Eq. 6)."""
    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D = torch.diag(deg_inv_sqrt)
    return D @ A @ D


def build_homogeneous_features(graph: HeteroGraph, device: torch.device) -> torch.Tensor:
    """
    Concatenates all node features into a single H^(0) [N, feat_dim].
    Since node types have different feature dims, we zero-pad all to max_dim.

    Layout: rows 0..N_P-1 = products, N_P..N_P+N_U-1 = users, N_P+N_U.. = sellers
    """
    xs = [graph.x_dict['product'], graph.x_dict['user'], graph.x_dict['seller']]
    max_dim = max(x.shape[1] for x in xs)
    padded  = []
    for x in xs:
        if x.shape[1] < max_dim:
            pad = torch.zeros(x.shape[0], max_dim - x.shape[1])
            x   = torch.cat([x, pad], dim=1)
        padded.append(x)
    H = torch.cat(padded, dim=0).to(device)   # [N, max_dim]
    return H


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    top_k: int = 20,
) -> Dict[str, float]:
    """
    Evaluates anomaly detection performance (paper Section V-3).

    Metrics:
      AUC-ROC    — overall discrimination
      AUC-PR     — precision-recall curve area (better for imbalanced)
      F1-score   — harmonic mean at threshold
      Precision@K / Recall@K — top-K ranked nodes
    """
    if labels.sum() == 0:
        return {'auc_roc': 0.0, 'auc_pr': 0.0, 'f1': 0.0,
                'prec_at_k': 0.0, 'recall_at_k': 0.0}

    metrics = {}

    # AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(labels, scores)
    except ValueError:
        metrics['auc_roc'] = 0.0

    # AUC-PR
    try:
        metrics['auc_pr'] = average_precision_score(labels, scores)
    except ValueError:
        metrics['auc_pr'] = 0.0

    # F1 at threshold
    preds = (scores >= threshold).astype(int)
    metrics['f1'] = f1_score(labels, preds, zero_division=0)
    metrics['precision'] = precision_score(labels, preds, zero_division=0)
    metrics['recall']    = recall_score(labels, preds, zero_division=0)

    # Precision@K / Recall@K
    k = min(top_k, len(scores))
    top_k_idx = np.argsort(scores)[::-1][:k]
    true_positives_at_k = labels[top_k_idx].sum()
    metrics['prec_at_k']   = true_positives_at_k / k
    metrics['recall_at_k'] = true_positives_at_k / max(labels.sum(), 1)

    return metrics


def move_adj_dict_to_device(
    adj_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in adj_dict.items()}
