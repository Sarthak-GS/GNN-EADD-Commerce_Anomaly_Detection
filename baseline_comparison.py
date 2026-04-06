"""
baseline_comparison.py  —  PyTorch Geometric Reference Baseline (Phase 1)

Implements the SAME dual-stage type-specific architecture as our custom
GNN-EADD, but using PyG ops as the '3rd-party GPU framework' reference.

Stage 1: Type-specific GAE — one RGCNConv encoder per edge type (mirrors
         our TypeSpecificGCNLayer) + inner-product decoder + BCE loss.
Stage 2: Type-specific GAT — one GATConv per edge type (mirrors our
         TypeSpecificGATLayer) + same combined loss (L_sup + λ L_unsup).

Because the architecture and hyper-params are identical, results should
closely match our custom implementation (small differences come only from
PyG's optimised sparse kernels vs. our dense matmul, and minor numerical
precision differences).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path


from utils.utils import (
    set_seed, build_homogeneous_features, compute_metrics,
)
from train import EDGE_TYPES

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# ── Hyper-params (must match train.py) ───────────────────────────────────────
EMBED_DIM  = 128
HIDDEN_DIM = 64
GAE_EPOCHS = 200
GAT_EPOCHS = 100
LR         = 1e-3
LAM        = 0.5   # λ for L_sup + λ·L_unsup


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build per-type sparse edge_index tensors in global node space
# ─────────────────────────────────────────────────────────────────────────────

def _build_edge_indices(graph):
    """
    Returns a dict  {rel_name -> edge_index [2, E]}  in global node space,
    plus a combined edge_index for the full graph.
    """
    N_P = graph.num_nodes_per_type['product']
    N_U = graph.num_nodes_per_type['user']

    off = {'product': 0, 'user': N_P, 'seller': N_P + N_U}

    rel_map = {
        ('product', 'purchase', 'user'): 'purchase',
        ('seller',  'sell',    'product'): 'sell',
        ('user',    'interact', 'user'): 'interact',
    }

    per_type = {}
    all_parts = []

    for (src_t, rel, dst_t), name in rel_map.items():
        ei = graph.edge_index_dict.get((src_t, rel, dst_t),
                                       torch.zeros(2, 0, dtype=torch.long))
        if ei.numel() == 0:
            per_type[name] = torch.zeros(2, 0, dtype=torch.long)
            continue
        src = ei[0] + off[src_t]
        dst = ei[1] + off[dst_t]
        fwd = torch.stack([src, dst], dim=0)
        bwd = torch.stack([dst, src], dim=0)
        per_type[name] = torch.cat([fwd, bwd], dim=1)   # undirected
        all_parts.append(per_type[name])

    # self-loops for 'self' relation
    N = sum(graph.num_nodes_per_type.values())
    idx = torch.arange(N)
    per_type['self'] = torch.stack([idx, idx], dim=0)
    all_parts.append(per_type['self'])

    combined = torch.cat(all_parts, dim=1) if all_parts else torch.zeros(2, 0, dtype=torch.long)
    return per_type, combined


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — PyG Type-Specific GAE
# ─────────────────────────────────────────────────────────────────────────────

class PyGTypeSpecificEncoder(nn.Module):
    """
    Two-layer encoder with one GCNConv per edge type (mirrors TypeSpecificGCNLayer).
    Each relation r has its own weight matrix W_r; messages are summed.
    """
    def __init__(self, in_dim, hidden_dim, embed_dim, edge_types):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.edge_types = edge_types
        # Layer 1
        self.W1 = nn.ModuleDict({r: GCNConv(in_dim,    hidden_dim, add_self_loops=False)
                                  for r in edge_types})
        # Layer 2
        self.W2 = nn.ModuleDict({r: GCNConv(hidden_dim, embed_dim,  add_self_loops=False)
                                  for r in edge_types})

    def forward(self, x, per_type_ei):
        # Layer 1: sum contributions across relation types
        h1 = sum(self.W1[r](x, per_type_ei[r])
                 for r in self.edge_types if per_type_ei[r].numel() > 0)
        h1 = F.relu(h1)

        # Layer 2
        z = sum(self.W2[r](h1, per_type_ei[r])
                for r in self.edge_types if per_type_ei[r].numel() > 0)
        return z   # [N, embed_dim]


def _inner_product_decode(z):
    """Â = σ(Z Zᵀ) — same as our GAEDecoder."""
    return torch.sigmoid(z @ z.t())


def _recon_loss(A, A_hat):
    """Same weighted BCE as our GraphAutoEncoder.reconstruction_loss."""
    n_pos = A.sum()
    n_neg = A.numel() - n_pos
    pos_weight = (n_neg / (n_pos + 1e-8)).clamp(max=10.0)
    return F.binary_cross_entropy_with_logits(
        A_hat.logit(eps=1e-6), A,
        pos_weight=torch.as_tensor(float(pos_weight), device=A.device),
        reduction='mean',
    )


def _build_full_adj(graph, device):
    """Dense [N,N] adjacency with self-loops — same as build_global_adjacency."""
    from data.graph_builder import build_global_adjacency
    return build_global_adjacency(graph).to(device)


def train_pyg_stage1(H, A_dense, per_type_ei, device):
    encoder = PyGTypeSpecificEncoder(
        H.shape[1], HIDDEN_DIM, EMBED_DIM, EDGE_TYPES
    ).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=LR)

    print(f"  [PyG] Stage 1: Training type-specific GAE ({GAE_EPOCHS} epochs)...")
    start_time = time.time()
    encoder.train()
    for ep in range(1, GAE_EPOCHS + 1):
        opt.zero_grad()
        Z     = encoder(H, per_type_ei)
        A_hat = _inner_product_decode(Z)
        loss  = _recon_loss(A_dense, A_hat)
        loss.backward()
        opt.step()
        if ep % 40 == 0 or ep == 1:
            print(f"    [GAE E{ep:>4}]  recon_loss = {loss.item():.6f}")

    encoder.eval()
    with torch.no_grad():
        Z_frozen = encoder(H, per_type_ei).detach()
        
    train_time = time.time() - start_time
    print(f"  [PyG] Stage 1 done in {train_time:.3f}s. Z shape: {Z_frozen.shape}")
    return Z_frozen, train_time


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — PyG Type-Specific GAT
# ─────────────────────────────────────────────────────────────────────────────

class PyGTypeSpecificGAT(nn.Module):
    """
    Two GAT layers each with per-edge-type GATConv (mirrors TypeSpecificGATLayer).
    Final sigmoid head → anomaly score in [0,1].
    """
    def __init__(self, embed_dim, hidden_dim, edge_types):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.edge_types = edge_types
        # Layer 1
        self.gat1 = nn.ModuleDict({r: GATConv(embed_dim,  hidden_dim, heads=1, add_self_loops=False)
                                    for r in edge_types})
        # Layer 2
        self.gat2 = nn.ModuleDict({r: GATConv(hidden_dim, hidden_dim, heads=1, add_self_loops=False)
                                    for r in edge_types})
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, z, per_type_ei):
        h1 = sum(self.gat1[r](z, per_type_ei[r])
                 for r in self.edge_types if per_type_ei[r].numel() > 0)
        h1 = F.relu(h1)

        h2 = sum(self.gat2[r](h1, per_type_ei[r])
                 for r in self.edge_types if per_type_ei[r].numel() > 0)
        h2 = F.relu(h2)

        return torch.sigmoid(self.score_head(h2)).squeeze(-1)  # [N]


def _sup_loss(s, y, mask):
    """Same weighted BCE as GraphAttentionNetwork.supervised_loss."""
    s_l, y_l = s[mask], y[mask].float()
    if s_l.numel() == 0:
        return torch.tensor(0.0, device=s.device)
    n_pos = y_l.sum().clamp(min=1)
    n_neg = y_l.numel() - n_pos
    w = (n_neg / n_pos).clamp(max=10.0)
    return F.binary_cross_entropy(
        s_l, y_l,
        weight=torch.where(y_l == 1, w * torch.ones_like(y_l), torch.ones_like(y_l)),
        reduction='mean',
    )


def _unsup_loss(s, A_dense):
    """Same smoothness loss as GraphAttentionNetwork.unsupervised_loss."""
    diff = s.unsqueeze(1) - s.unsqueeze(0)
    return (A_dense * diff**2).sum() / (A_dense.sum() + 1e-8)


def train_pyg_stage2(Z_frozen, A_dense, per_type_ei, y, labeled_mask, device):
    model = PyGTypeSpecificGAT(EMBED_DIM, HIDDEN_DIM, EDGE_TYPES).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"  [PyG] Stage 2: Training type-specific GAT ({GAT_EPOCHS} epochs)...")
    start_time = time.time()
    model.train()
    for ep in range(1, GAT_EPOCHS + 1):
        opt.zero_grad()
        s = model(Z_frozen, per_type_ei)
        l_sup   = _sup_loss(s, y, labeled_mask)
        l_unsup = _unsup_loss(s, A_dense)
        loss    = l_sup + LAM * l_unsup
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == 1:
            print(f"    [GAT E{ep:>4}]  loss={loss.item():.5f}  "
                  f"sup={l_sup.item():.5f}  unsup={l_unsup.item():.5f}")

    model.eval()
    with torch.no_grad():
        scores = model(Z_frozen, per_type_ei).cpu().numpy()
        
    train_time = time.time() - start_time
    print(f"  [PyG] Stage 2 done in {train_time:.3f}s.")
    return scores, train_time


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_comparison(graph_path: str = 'data/graph.pt'):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Baselines] Device: {device}")

    path = Path(graph_path)
    if not path.exists():
        print(f"[!] Graph file not found: {graph_path}. Please run generate_data.py first.")
        return

    graph = torch.load(path)
    H     = build_homogeneous_features(graph, device)
    y     = graph.y.to(device)
    labeled_mask = graph.labeled_mask.to(device)
    test_mask    = graph.test_mask

    per_type_ei, _ = _build_edge_indices(graph)
    per_type_ei = {r: ei.to(device) for r, ei in per_type_ei.items()}
    A_dense = _build_full_adj(graph, device)

    # ── PyG Dual-Stage (Type-Specific, mirrors our architecture) ─────────────
    print("\n" + "="*60)
    print("  BASELINE: PyTorch Geometric Dual-Stage Type-Specific")
    print("="*60)

    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        print("  [!] PyG not installed. Run:  pip install torch-geometric")
        return

    Z_frozen, pyg_gae_time = train_pyg_stage1(H, A_dense, per_type_ei, device)
    scores, pyg_gat_time   = train_pyg_stage2(Z_frozen, A_dense, per_type_ei, y, labeled_mask, device)

    # Evaluate on same held-out test split
    test_np   = test_mask.numpy()
    y_np      = graph.y.numpy()
    metrics_pyg = compute_metrics(scores[test_np], y_np[test_np])
    metrics_pyg['gae_time'] = pyg_gae_time
    metrics_pyg['gat_time'] = pyg_gat_time
    
    print(f"\n  [PyG Test] AUC-ROC: {metrics_pyg['auc_roc']:.4f}  "
          f"AUC-PR: {metrics_pyg['auc_pr']:.4f}  F1: {metrics_pyg['f1']:.4f}")

    # ── Comparison Table ─────────────────────────────────────────────────────
    ckpt_path = RESULTS_DIR / 'checkpoint.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        metrics_ours = ckpt.get('metrics_test', {})

        print("\n" + "="*60)
        print("  COMPARISON TABLE  (evaluated on same held-out test split)")
        print("="*60)
        print(f"  {'Method':<36} {'AUC-ROC':>9} {'AUC-PR':>8} {'F1':>8}")
        print(f"  {'-'*64}")
        print(f"  {'PyG Type-Specific (3rd-party GPU)':<36} "
              f"{metrics_pyg['auc_roc']:>9.4f} {metrics_pyg['auc_pr']:>8.4f} "
              f"{metrics_pyg['f1']:>8.4f}")
        print(f"  {'GNN-EADD custom (ours)':<36} "
              f"{metrics_ours['auc_roc']:>9.4f} {metrics_ours['auc_pr']:>8.4f} "
              f"{metrics_ours['f1']:>8.4f}")
        print(f"  {'='*64}")
        print()
        print("  Note: Small differences expected from sparse vs dense matmul")
        print("  and PyG's internal normalisation. Phase 2 CUDA kernels will")
        print("  speed up our implementation without changing these numbers.")
    else:
        print(f"\n[!] Run train.py first to see GNN-EADD results.")

    np.save(RESULTS_DIR / 'baseline_pyg_metrics.npy', metrics_pyg)
    print(f"\n[Saved] PyG baseline metrics -> {RESULTS_DIR}/baseline_pyg_metrics.npy")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--graph_path', type=str, default='data/graph.pt')
    args = p.parse_args()
    run_baseline_comparison(args.graph_path)
