import torch
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class HeteroGraph:
    """
    Container for the full heterogeneous graph.
    It's heterogeneous, just like my music taste :)
    """
    x_dict: Dict[str, torch.Tensor]
    edge_index_dict: Dict[Tuple, torch.Tensor]
    y: torch.Tensor
    node_type_bounds: Dict[str, Tuple[int, int]]
    num_nodes_per_type: Dict[str, int]
    labeled_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

def _normalize_features(x: np.ndarray) -> np.ndarray:
    col_min, col_max = x.min(axis=0, keepdims=True), x.max(axis=0, keepdims=True)
    denom = (col_max - col_min)
    denom[denom == 0] = 1.0
    return (x - col_min) / denom

def generate_synthetic_graph(n_products=200, n_users=80, n_sellers=20, anomaly_fraction=0.15, seed=42):
    """ Generating a synthetic world because reality is too complicated :( """
    rng = np.random.default_rng(seed)
    random.seed(seed); torch.manual_seed(seed)
    
    xp = rng.uniform(0.2, 0.8, (n_products, 8)).astype(np.float32)
    anom_p = rng.choice(n_products, max(1, int(anomaly_fraction * n_products)), replace=False)
    xp[anom_p, 0] = rng.uniform(0.10, 0.35, len(anom_p))
    xp = np.clip(xp + rng.normal(0, 0.08, xp.shape), 0, 1)

    xu = np.clip(rng.uniform(0.2, 0.8, (n_users, 6)) + rng.normal(0, 0.08, (n_users, 6)), 0, 1)
    xs = np.clip(rng.uniform(0.2, 0.8, (n_sellers, 5)) + rng.normal(0, 0.08, (n_sellers, 5)), 0, 1)

    y = torch.zeros(n_products + n_users + n_sellers, dtype=torch.long)
    y[anom_p] = 1
    
    # Simple purchase edges
    p_indices = rng.choice(n_products, n_products * 2)
    u_indices = rng.choice(n_users, n_products * 2)
    ei_pu = torch.stack([torch.from_numpy(p_indices), torch.from_numpy(u_indices)], dim=0).long()

    # Simple sell edges
    s_indices = rng.choice(n_sellers, n_products)
    p_indices_s = torch.arange(n_products)
    ei_sp = torch.stack([torch.from_numpy(s_indices), p_indices_s], dim=0).long()

    idx = rng.permutation(len(y))
    m_train, m_val, m_test = torch.zeros_like(y, dtype=torch.bool), torch.zeros_like(y, dtype=torch.bool), torch.zeros_like(y, dtype=torch.bool)
    m_train[idx[:int(0.6*len(y))]] = True
    m_val[idx[int(0.6*len(y)):int(0.8*len(y))]] = True
    m_test[idx[int(0.8*len(y)):]] = True

    return HeteroGraph(
        x_dict={'product': torch.from_numpy(xp), 'user': torch.from_numpy(xu), 'seller': torch.from_numpy(xs)},
        edge_index_dict={('product', 'purchase', 'user'): ei_pu, ('seller', 'sell', 'product'): ei_sp},
        y=y,
        node_type_bounds={'product': (0, n_products), 'user': (n_products, n_products+n_users), 'seller': (n_products+n_users, n_products+n_users+n_sellers)},
        num_nodes_per_type={'product': n_products, 'user': n_users, 'seller': n_sellers},
        labeled_mask=m_train, val_mask=m_val, test_mask=m_test
    )

def build_global_adjacency(graph: HeteroGraph) -> torch.Tensor:
    N = sum(graph.num_nodes_per_type.values())
    A = torch.eye(N)
    offset = {'product': 0, 'user': graph.num_nodes_per_type['product'], 'seller': graph.num_nodes_per_type['product'] + graph.num_nodes_per_type['user']}
    
    for (src_t, rel, dst_t), ei in graph.edge_index_dict.items():
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            g_s, g_d = offset[src_t] + s, offset[dst_t] + d
            A[g_s, g_d] = A[g_d, g_s] = 1.0
    return A
