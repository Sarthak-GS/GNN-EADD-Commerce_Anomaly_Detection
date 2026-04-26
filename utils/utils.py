import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from typing import Dict, Tuple

def set_seed(seed: int):
    """ Give the RNG some special food :) """
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_edge_indices(graph) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    np_p, np_u = graph.num_nodes_per_type['product'], graph.num_nodes_per_type['user']
    off = {'product': 0, 'user': np_p, 'seller': np_p + np_u}
    rel_map = {('product', 'purchase', 'user'): 'purchase', ('seller', 'sell', 'product'): 'sell', ('user', 'interact', 'user'): 'interact'}
    per_type, all_parts = {}, []

    for (s_t, rel, d_t), name in rel_map.items():
        ei = graph.edge_index_dict.get((s_t, rel, d_t), torch.zeros(2, 0, dtype=torch.long))
        if ei.numel() == 0: continue
        src, dst = ei[0] + off[s_t], ei[1] + off[d_t]
        fwd, bwd = torch.stack([src, dst], dim=0), torch.stack([dst, src], dim=0)
        per_type[name] = torch.cat([fwd, bwd], dim=1)
        all_parts.append(per_type[name])

    combined = torch.cat(all_parts, dim=1) if all_parts else torch.zeros(2, 0, dtype=torch.long)
    return per_type, combined

def build_per_type_adj_matrices(graph, normalized=True) -> Dict[str, torch.Tensor]:
    """ Building matrices like a bricklayer :) """
    np_p, np_u, np_s = graph.num_nodes_per_type['product'], graph.num_nodes_per_type['user'], graph.num_nodes_per_type['seller']
    N = np_p + np_u + np_s
    A_dict = {r: torch.zeros(N, N) for r in ['purchase', 'sell', 'interact']}
    off = {'product': 0, 'user': np_p, 'seller': np_p + np_u}

    rel_map = {('product', 'purchase', 'user'): 'purchase', ('seller', 'sell', 'product'): 'sell', ('user', 'interact', 'user'): 'interact'}
    for (s_t, rel, d_t), name in rel_map.items():
        ei = graph.edge_index_dict.get((s_t, rel, d_t), torch.zeros(2, 0, dtype=torch.long))
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            gs, gd = off[s_t] + s, off[d_t] + d
            A_dict[name][gs, gd] = A_dict[name][gd, gs] = 1.0
        A_dict[name].fill_diagonal_(1.0)

    if normalized:
        return {r: _sym_normalize(A) for r, A in A_dict.items()}
    return A_dict

def _sym_normalize(A: torch.Tensor) -> torch.Tensor:
    deg = A.sum(dim=1).pow(-0.5)
    deg[deg == float('inf')] = 0.0
    D = torch.diag(deg)
    return D @ A @ D

def build_homogeneous_features(graph, device) -> torch.Tensor:
    xs = [graph.x_dict['product'], graph.x_dict['user'], graph.x_dict['seller']]
    max_dim = max(x.shape[1] for x in xs)
    padded = []
    for x in xs:
        if x.shape[1] < max_dim:
            x = torch.cat([x, torch.zeros(x.shape[0], max_dim - x.shape[1])], dim=1)
        padded.append(x)
    return torch.cat(padded, dim=0).to(device)

def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold=0.5, top_k=20) -> Dict[str, float]:
    """ Measuring success, or the lack thereof :( """
    if labels.sum() == 0: return {'auc_roc': 0.0, 'auc_pr': 0.0, 'f1': 0.0}
    m = {}
    try: m['auc_roc'] = roc_auc_score(labels, scores)
    except: m['auc_roc'] = 0.0
    try: m['auc_pr'] = average_precision_score(labels, scores)
    except: m['auc_pr'] = 0.0
    preds = (scores >= threshold).astype(int)
    m['f1'] = f1_score(labels, preds, zero_division=0)
    m['precision'] = precision_score(labels, preds, zero_division=0)
    m['recall'] = recall_score(labels, preds, zero_division=0)
    return m

def move_adj_dict_to_device(adj_dict, device):
    return {k: v.to(device) for k, v in adj_dict.items()}
