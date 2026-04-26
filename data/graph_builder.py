"""
graph_builder.py
Builds the heterogeneous e-commerce graph G = (V, E, phi, psi).
Node types: product (P), user (U), seller (S)
Edge types: purchase (P-U), sell (S-P), interact (U-U)
"""

import torch
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class HeteroGraph:
    """Container for the full heterogeneous graph.

    Attributes:
        x_dict  : {node_type -> feature tensor [N_type, feat_dim]}
        edge_index_dict : {(src_type, rel, dst_type) -> edge_index [2, E]}
        y       : label tensor [N_total]  (0=normal, 1=anomaly)
        node_type_bounds : maps node type -> (start_idx, end_idx) in global id space
        num_nodes_per_type : {type -> count}
    """
    x_dict: Dict[str, torch.Tensor]
    edge_index_dict: Dict[Tuple, torch.Tensor]
    y: torch.Tensor
    node_type_bounds: Dict[str, Tuple[int, int]]
    num_nodes_per_type: Dict[str, int]
    labeled_mask: torch.Tensor    # True for nodes in TRAINING split
    val_mask: torch.Tensor         # True for nodes in VALIDATION split
    test_mask: torch.Tensor        # True for nodes in TEST split (both classes present)


def _normalize_features(x: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1] per feature column."""
    col_min = x.min(axis=0, keepdims=True)
    col_max = x.max(axis=0, keepdims=True)
    denom = col_max - col_min
    denom[denom == 0] = 1.0   # avoid div-by-zero for constant columns
    return (x - col_min) / denom


def generate_synthetic_graph(
    n_products: int = 200,
    n_users: int = 80,
    n_sellers: int = 20,
    product_feat_dim: int = 8,
    user_feat_dim: int = 6,
    seller_feat_dim: int = 5,
    anomaly_fraction: float = 0.15,
    labeled_fraction: float = 0.60,
    seed: int = 42,
) -> HeteroGraph:
    """
    Creates a synthetic heterogeneous e-commerce graph for testing/demo.

    Graph topology (from paper Sec IV-A):
      - Products p connect to users  u  via 'purchase' edges (Epu)
      - Products p connect to sellers s  via 'sell'     edges (Eps)
      - Users    u connect to users  u  via 'interact'  edges (Euu)

    Anomaly injection strategy:
      - Anomalous nodes get SUBTLY different features (overlapping with normal range)
      - Anomalous nodes form a NOISY community (not perfectly separated)
      - Cross-community edges exist to prevent trivial structural detection
      - The combination of features + structure is needed for detection

    Returns HeteroGraph ready for training.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    N_P, N_U, N_S = n_products, n_users, n_sellers
    N_total = N_P + N_U + N_S

    # ── 1. NODE FEATURES ────────────────────────────────────────────────────────
    # Product features: [price, avg_rating, review_count, category_emb×5]
    n_anomaly_products = max(1, int(anomaly_fraction * N_P))
    n_anomaly_users    = max(1, int(anomaly_fraction * N_U))
    n_anomaly_sellers  = max(1, int(anomaly_fraction * N_S))

    # --- Products ---
    # Normal products: features drawn from [0.2, 0.8]
    xp = rng.uniform(0.2, 0.8, (N_P, product_feat_dim)).astype(np.float32)
    anomaly_product_idx = rng.choice(N_P, n_anomaly_products, replace=False)
    # Anomalous products: SUBTLE shift — ranges OVERLAP with normal
    # price slightly lower, rating slightly higher, but not extreme
    xp[anomaly_product_idx, 0] = rng.uniform(0.10, 0.35, n_anomaly_products)  # cheaper
    xp[anomaly_product_idx, 1] = rng.uniform(0.65, 0.85, n_anomaly_products)  # higher rating
    xp[anomaly_product_idx, 2] = rng.uniform(0.55, 0.80, n_anomaly_products)  # more reviews
    # Add noise to ALL features so they're not clean signals
    xp += rng.normal(0, 0.08, xp.shape).astype(np.float32)
    xp = np.clip(xp, 0, 1)

    # --- Users ---
    xu = rng.uniform(0.2, 0.8, (N_U, user_feat_dim)).astype(np.float32)
    anomaly_user_idx = rng.choice(N_U, n_anomaly_users, replace=False)
    xu[anomaly_user_idx, 0] = rng.uniform(0.60, 0.85, n_anomaly_users)  # higher purchase freq
    xu[anomaly_user_idx, 2] = rng.uniform(0.60, 0.85, n_anomaly_users)  # higher review rate
    xu += rng.normal(0, 0.08, xu.shape).astype(np.float32)
    xu = np.clip(xu, 0, 1)

    # --- Sellers ---
    xs = rng.uniform(0.2, 0.8, (N_S, seller_feat_dim)).astype(np.float32)
    anomaly_seller_idx = rng.choice(N_S, n_anomaly_sellers, replace=False)
    xs[anomaly_seller_idx, 0] = rng.uniform(0.15, 0.40, n_anomaly_sellers)  # lower rep
    xs[anomaly_seller_idx, 2] = rng.uniform(0.55, 0.75, n_anomaly_sellers)  # higher return rate
    xs += rng.normal(0, 0.08, xs.shape).astype(np.float32)
    xs = np.clip(xs, 0, 1)

    # Normalize all features to [0,1]
    xp = _normalize_features(xp)
    xu = _normalize_features(xu)
    xs = _normalize_features(xs)

    # ── 2. GLOBAL LABELS ────────────────────────────────────────────────────────
    labels = np.zeros(N_total, dtype=np.int64)
    labels[anomaly_product_idx]               = 1
    labels[N_P + anomaly_user_idx]            = 1
    labels[N_P + N_U + anomaly_seller_idx]    = 1

    # ── 3. EDGE CONSTRUCTION (with noise) ────────────────────────────────────
    normal_product_list = sorted(set(range(N_P)) - set(anomaly_product_idx.tolist()))
    normal_user_list    = sorted(set(range(N_U)) - set(anomaly_user_idx.tolist()))
    normal_seller_list  = sorted(set(range(N_S)) - set(anomaly_seller_idx.tolist()))

    # Purchase edges: user -> product (each user buys 2-5 products)
    # KEY: anomalous users buy anomalous products with only 50% probability (noisy!)
    # And 15% of normal users also buy an anomalous product (cross-community noise)
    pu_src, pu_dst = [], []   # product_idx, user_idx (local space)

    for u in range(N_U):
        n_purchases = int(rng.integers(2, 6))
        if u in anomaly_user_idx:
            # Anomalous user: 50% chance to buy anomalous product per purchase
            for _ in range(n_purchases):
                if rng.random() < 0.50 and len(anomaly_product_idx) > 0:
                    pu_src.append(int(rng.choice(anomaly_product_idx)))
                else:
                    pu_src.append(int(rng.choice(normal_product_list or [0])))
                pu_dst.append(u)
        else:
            # Normal user: mostly buys normal products, but 15% noise
            for _ in range(n_purchases):
                if rng.random() < 0.15 and len(anomaly_product_idx) > 0:
                    pu_src.append(int(rng.choice(anomaly_product_idx)))
                else:
                    pu_src.append(int(rng.choice(normal_product_list or [0])))
                pu_dst.append(u)

    # Sell edges: seller -> product
    # Anomalous products sold by anomalous sellers 70% of time (not 100%!)
    # Some normal sellers also sell anomalous products (noise)
    sp_src, sp_dst = [], []
    for p in range(N_P):
        if p in anomaly_product_idx:
            if rng.random() < 0.70 and len(anomaly_seller_idx) > 0:
                s = int(rng.choice(anomaly_seller_idx))
            else:
                s = int(rng.choice(normal_seller_list or [0]))
        else:
            if rng.random() < 0.05 and len(anomaly_seller_idx) > 0:
                # 5% of normal products sold by anomalous sellers (noise)
                s = int(rng.choice(anomaly_seller_idx))
            else:
                s = int(rng.choice(normal_seller_list or [0]))
        sp_src.append(s)
        sp_dst.append(p)

    # User-user interaction edges (co-purchasing patterns)
    # Scalable sampling approach (O(N_U) instead of O(N_U^2))
    anomaly_user_set = set(anomaly_user_idx.tolist())
    uu_src, uu_dst = [], []

    for i in range(N_U):
        is_anom_i = i in anomaly_user_set
        # Degree sampling: anomalous users have more connections
        degree = int(rng.poisson(lam=8 if is_anom_i else 5))
        degree = max(1, min(degree, N_U - 1))

        # Randomly sample neighbors
        candidates = rng.choice(N_U, size=degree, replace=False)
        for j in candidates:
            if i == j: continue
            
            # Keep edge with higher probability if both are anomalous
            both_anom = is_anom_i and (j in anomaly_user_set)
            if both_anom:
                prob = 0.25
            elif is_anom_i or (j in anomaly_user_set):
                prob = 0.06
            else:
                prob = 0.04
                
            if rng.random() < prob:
                uu_src.extend([i, int(j)])
                uu_dst.extend([int(j), i])

    # ── 4. GLOBAL EDGE INDICES ──────────────────────────────────────────────
    edge_index_dict = {}

    if pu_src:
        edge_index_dict[('product', 'purchase', 'user')] = torch.tensor(
            [pu_src, pu_dst], dtype=torch.long)
    else:
        edge_index_dict[('product', 'purchase', 'user')] = torch.zeros(2, 0, dtype=torch.long)

    if sp_src:
        edge_index_dict[('seller', 'sell', 'product')] = torch.tensor(
            [sp_src, sp_dst], dtype=torch.long)
    else:
        edge_index_dict[('seller', 'sell', 'product')] = torch.zeros(2, 0, dtype=torch.long)

    if uu_src:
        edge_index_dict[('user', 'interact', 'user')] = torch.tensor(
            [uu_src, uu_dst], dtype=torch.long)
    else:
        edge_index_dict[('user', 'interact', 'user')] = torch.zeros(2, 0, dtype=torch.long)

    # ── 5. PROPER TRAIN / VAL / TEST SPLIT ──────────────────────────────────
    # Each split MUST contain both normal and anomalous nodes so AUC is computable.
    # Strategy: split anomalies 60/20/20, then sample matching normal nodes.

    anomaly_global_idx = np.where(labels == 1)[0]
    normal_global_idx  = np.where(labels == 0)[0]

    rng.shuffle(anomaly_global_idx)
    n_a = len(anomaly_global_idx)
    n_train_a = max(2, int(0.60 * n_a))
    n_val_a   = max(2, int(0.20 * n_a))
    n_test_a  = max(2, n_a - n_train_a - n_val_a)
    # Adjust if we overallocated
    while n_train_a + n_val_a + n_test_a > n_a:
        n_train_a = max(1, n_train_a - 1)

    train_anom = anomaly_global_idx[:n_train_a]
    val_anom   = anomaly_global_idx[n_train_a:n_train_a + n_val_a]
    test_anom  = anomaly_global_idx[n_train_a + n_val_a:n_train_a + n_val_a + n_test_a]

    # Normal nodes: split proportionally
    rng.shuffle(normal_global_idx)
    n_n = len(normal_global_idx)
    n_train_n = max(3, int(0.60 * n_n))
    n_val_n   = max(3, int(0.20 * n_n))
    n_test_n  = max(3, n_n - n_train_n - n_val_n)
    while n_train_n + n_val_n + n_test_n > n_n:
        n_train_n = max(1, n_train_n - 1)

    train_norm = normal_global_idx[:n_train_n]
    val_norm   = normal_global_idx[n_train_n:n_train_n + n_val_n]
    test_norm  = normal_global_idx[n_train_n + n_val_n:n_train_n + n_val_n + n_test_n]

    # Build masks
    labeled_mask = torch.zeros(N_total, dtype=torch.bool)
    val_mask     = torch.zeros(N_total, dtype=torch.bool)
    test_mask    = torch.zeros(N_total, dtype=torch.bool)

    train_idx = np.concatenate([train_anom, train_norm])
    val_idx   = np.concatenate([val_anom,   val_norm])
    test_idx  = np.concatenate([test_anom,  test_norm])

    labeled_mask[train_idx] = True
    val_mask[val_idx]       = True
    test_mask[test_idx]     = True

    print(f"[Split] Train: {labeled_mask.sum().item()} nodes "
          f"(anom={len(train_anom)}, norm={len(train_norm)}) | "
          f"Val: {val_mask.sum().item()} (anom={len(val_anom)}, norm={len(val_norm)}) | "
          f"Test: {test_mask.sum().item()} (anom={len(test_anom)}, norm={len(test_norm)})")

    node_type_bounds = {
        'product': (0, N_P),
        'user':    (N_P, N_P + N_U),
        'seller':  (N_P + N_U, N_total),
    }

    graph = HeteroGraph(
        x_dict={
            'product': torch.tensor(xp, dtype=torch.float),
            'user':    torch.tensor(xu, dtype=torch.float),
            'seller':  torch.tensor(xs, dtype=torch.float),
        },
        edge_index_dict=edge_index_dict,
        y=torch.tensor(labels, dtype=torch.long),
        node_type_bounds=node_type_bounds,
        num_nodes_per_type={'product': N_P, 'user': N_U, 'seller': N_S},
        labeled_mask=labeled_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    print(f"[Graph] Nodes: P={N_P}, U={N_U}, S={N_S}, total={N_total}")
    print(f"[Graph] Anomalies: P={n_anomaly_products}, U={n_anomaly_users}, S={n_anomaly_sellers} "
          f"(total={n_a})")
    print(f"[Graph] Edges: purchase={len(pu_src)}, sell={len(sp_src)}, interact={len(uu_src)}")
    return graph


def build_global_adjacency(graph: HeteroGraph) -> torch.Tensor:
    """
    Builds the full N_total x N_total adjacency matrix A (with self-loops).
    Used by Stage 1 GAE decoder to compute reconstruction loss.
    """
    N_P = graph.num_nodes_per_type['product']
    N_U = graph.num_nodes_per_type['user']
    N_S = graph.num_nodes_per_type['seller']
    N   = N_P + N_U + N_S

    A = torch.zeros(N, N, dtype=torch.float)

    # Self-loops for all nodes (paper Section IV-A step 4)
    A.fill_diagonal_(1.0)

    # Purchase edges: product i ↔ user j  (symmetric)
    ei = graph.edge_index_dict.get(('product', 'purchase', 'user'), torch.zeros(2, 0, dtype=torch.long))
    for p_idx, u_idx in zip(ei[0].tolist(), ei[1].tolist()):
        g_p = p_idx
        g_u = N_P + u_idx
        A[g_p, g_u] = 1.0
        A[g_u, g_p] = 1.0

    # Sell edges: seller i ↔ product j
    ei = graph.edge_index_dict.get(('seller', 'sell', 'product'), torch.zeros(2, 0, dtype=torch.long))
    for s_idx, p_idx in zip(ei[0].tolist(), ei[1].tolist()):
        g_s = N_P + N_U + s_idx
        g_p = p_idx
        A[g_s, g_p] = 1.0
        A[g_p, g_s] = 1.0

    # User-user interaction edges
    ei = graph.edge_index_dict.get(('user', 'interact', 'user'), torch.zeros(2, 0, dtype=torch.long))
    for u1, u2 in zip(ei[0].tolist(), ei[1].tolist()):
        g_u1 = N_P + u1
        g_u2 = N_P + u2
        A[g_u1, g_u2] = 1.0

    return A


def build_normalized_adj(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalized adjacency:  D^{-1/2} A D^{-1/2}
    This is the normalization used in GCN / GAE encoder (Eq. 6 in paper).
    """
    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


if __name__ == '__main__':
    g = generate_synthetic_graph()
    A = build_global_adjacency(g)
    A_norm = build_normalized_adj(A)
    print(f"A shape: {A.shape}, density: {A.sum().item() / (A.shape[0]**2):.3f}")
    print(f"A_norm: min={A_norm.min().item():.4f}, max={A_norm.max().item():.4f}")
    print(f"Feature shapes: { {k: v.shape for k,v in g.x_dict.items()} }")
