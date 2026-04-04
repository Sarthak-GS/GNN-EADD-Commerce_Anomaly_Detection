"""
train.py  —  Full GNN-EADD Training Pipeline (Algorithm 1 from paper)

This file implements:
  Stage 1: Train GAE for N_GAE epochs (unsupervised, no labels used)
  Stage 2: Freeze GAE, train GAT for N_GAT epochs (semi-supervised)

Hyperparameters (paper Section V-4):
  - embed_dim = 128
  - hidden_dim = 64
  - lr = 0.001 (Adam)
  - N_GAE = 200 epochs
  - N_GAT = 100 epochs
  - lambda = 0.5 (balance supervised / unsupervised loss)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from typing import Optional

from data.graph_builder import generate_synthetic_graph, build_global_adjacency
from models.gae import GraphAutoEncoder
from models.gat import GraphAttentionNetwork
from utils.utils import (
    set_seed,
    build_per_type_adj_matrices,
    build_homogeneous_features,
    compute_metrics,
    move_adj_dict_to_device,
)


EDGE_TYPES = ['purchase', 'sell', 'interact', 'self']


def select_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")
    return device


# ────────────────────────────────────────────────────────────────────────────────
# STAGE 1: GRAPH AUTOENCODER TRAINING
# ────────────────────────────────────────────────────────────────────────────────

def train_gae(
    gae: GraphAutoEncoder,
    H: torch.Tensor,                              # [N, feat_dim]  all node features
    A: torch.Tensor,                              # [N, N]         full adjacency
    A_norm_per_type: dict,                        # {rel -> [N,N]} normalized
    n_epochs: int = 200,
    lr: float = 1e-3,
    log_every: int = 20,
) -> list:
    """
    Trains GAE to minimize reconstruction loss (Eq. 9).
    Labels are NOT used — purely unsupervised.

    Returns list of loss values per epoch.
    """
    optimizer = optim.Adam(gae.parameters(), lr=lr)
    losses = []

    print(f"\n{'='*60}")
    print("  STAGE 1: GAE Unsupervised Training")
    print(f"{'='*60}")

    gae.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        Z, A_hat = gae(H, A_norm_per_type)
        loss = gae.reconstruction_loss(A, A_hat)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % log_every == 0 or epoch == 1:
            print(f"  [GAE E{epoch:>4}]  recon_loss = {loss.item():.6f}")

    print(f"  GAE training done. Final loss: {losses[-1]:.6f}")
    return losses


# ────────────────────────────────────────────────────────────────────────────────
# STAGE 2: GAT SEMI-SUPERVISED TRAINING
# ────────────────────────────────────────────────────────────────────────────────

def train_gat(
    gat: GraphAttentionNetwork,
    Z_init: torch.Tensor,                # [N, embed_dim]  frozen GAE embeddings
    A: torch.Tensor,                     # [N, N]          full adjacency
    A_per_type: dict,                    # {rel -> [N,N]}  for attention computation
    y: torch.Tensor,                     # [N]             all labels
    labeled_mask: torch.Tensor,          # [N] bool        which nodes are labeled
    n_epochs: int = 100,
    lr: float = 1e-3,
    lam: float = 0.5,
    log_every: int = 10,
) -> list:
    """
    Trains GAT on the frozen embeddings Z_init from Stage 1.
    Uses L_GAT = L_sup + λ * L_unsup  (Eq. 15).

    Returns list of (total_loss, sup_loss, unsup_loss) per epoch.
    """
    optimizer = optim.Adam(gat.parameters(), lr=lr)
    history = []

    print(f"\n{'='*60}")
    print("  STAGE 2: GAT Semi-supervised Training")
    print(f"{'='*60}")

    gat.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        s = gat(Z_init, A_per_type)         # [N] anomaly scores

        total_loss, l_sup, l_unsup = GraphAttentionNetwork.combined_loss(
            s=s, y=y, labeled_mask=labeled_mask, A=A, lam=lam
        )

        total_loss.backward()
        optimizer.step()

        history.append((total_loss.item(), l_sup.item(), l_unsup.item()))

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"  [GAT E{epoch:>4}]  loss={total_loss.item():.5f}  "
                f"sup={l_sup.item():.5f}  unsup={l_unsup.item():.5f}"
            )

    print(f"  GAT training done. Final loss: {history[-1][0]:.5f}")
    return history


# ────────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    gat: GraphAttentionNetwork,
    Z: torch.Tensor,
    A_per_type: dict,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    label: str = "Test",
) -> dict:
    """
    Runs inference and computes all evaluation metrics.
    mask: if provided, evaluate only on those nodes.
    """
    gat.eval()
    scores = gat(Z, A_per_type).cpu().numpy()   # [N]
    labels = y.cpu().numpy()

    if mask is not None:
        mask_np = mask.cpu().numpy()
        scores = scores[mask_np]
        labels = labels[mask_np]

    metrics = compute_metrics(scores, labels)
    print(f"\n[{label} Results]")
    print(f"  AUC-ROC    : {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR     : {metrics['auc_pr']:.4f}")
    print(f"  F1-score   : {metrics['f1']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  Prec@K     : {metrics['prec_at_k']:.4f}")
    print(f"  Recall@K   : {metrics['recall_at_k']:.4f}")
    return metrics


# ────────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP  (Algorithm 1)
# ────────────────────────────────────────────────────────────────────────────────

def run_training(args):
    set_seed(args.seed)
    device = select_device()

    # ── DATA ──────────────────────────────────────────────────────────────────
    print("\n[Data] Building synthetic heterogeneous e-commerce graph...")
    graph = generate_synthetic_graph(
        n_products=args.n_products,
        n_users=args.n_users,
        n_sellers=args.n_sellers,
        anomaly_fraction=args.anomaly_fraction,
        seed=args.seed,
    )

    # Feature matrix H^(0): [N, feat_dim]
    H = build_homogeneous_features(graph, device)

    # Full adjacency A [N, N]
    A = build_global_adjacency(graph).to(device)

    # Per-type normalized adjacencies for GAE encoder AND GAT attention
    A_norm_per_type = build_per_type_adj_matrices(graph, normalized=True)
    A_per_type      = build_per_type_adj_matrices(graph, normalized=False)
    A_norm_per_type = move_adj_dict_to_device(A_norm_per_type, device)
    A_per_type      = move_adj_dict_to_device(A_per_type, device)

    y             = graph.y.to(device)
    labeled_mask  = graph.labeled_mask.to(device)
    val_mask      = getattr(graph, 'val_mask', torch.zeros_like(labeled_mask)).to(device)
    test_mask     = graph.test_mask.to(device)

    feat_dim  = H.shape[1]
    N         = H.shape[0]

    print(f"\n[Config] N={N}, feat_dim={feat_dim}, embed_dim={args.embed_dim}, "
          f"hidden_dim={args.hidden_dim}")
    print(f"[Config] GAE epochs={args.gae_epochs}, GAT epochs={args.gat_epochs}, "
          f"lr={args.lr}, λ={args.lam}")

    # ── STAGE 1: GAE ──────────────────────────────────────────────────────────
    gae = GraphAutoEncoder(
        feat_dim   = feat_dim,
        hidden_dim = args.hidden_dim,
        embed_dim  = args.embed_dim,
        edge_types = EDGE_TYPES,
    ).to(device)

    gae_losses = train_gae(
        gae, H, A, A_norm_per_type,
        n_epochs  = args.gae_epochs,
        lr        = args.lr,
        log_every = max(1, args.gae_epochs // 10),
    )

    # Extract frozen embeddings Z (no gradient needed for Stage 2 input)
    gae.eval()
    with torch.no_grad():
        Z = gae.encode(H, A_norm_per_type).detach()  # [N, embed_dim]

    print(f"\n[Stage 1 done] Z shape: {Z.shape}, "
          f"Z range: [{Z.min().item():.4f}, {Z.max().item():.4f}]")

    # ── STAGE 2: GAT ──────────────────────────────────────────────────────────
    gat = GraphAttentionNetwork(
        embed_dim  = args.embed_dim,
        hidden_dim = args.hidden_dim,
        edge_types = EDGE_TYPES,
    ).to(device)

    gat_history = train_gat(
        gat, Z, A, A_per_type, y, labeled_mask,
        n_epochs  = args.gat_epochs,
        lr        = args.lr,
        lam       = args.lam,
        log_every = max(1, args.gat_epochs // 10),
    )

    # ── EVALUATION ────────────────────────────────────────────────────────────
    # Prevent data leakage: ONLY evaluate on held-out test nodes
    print("\n" + "="*60)
    print("  STAGE 3: Evaluation (on held-out splits)")
    print("="*60)
    
    if val_mask.any():
        metrics_val = evaluate(gat, Z, A_per_type, y, mask=val_mask, label="Validation")
    
    if test_mask.any():
        metrics_test = evaluate(gat, Z, A_per_type, y, mask=test_mask, label="Test (Unseen)")
    else:
        print("[!] Warning: No test mask found, evaluating on all nodes (contains leakage)")
        metrics_test = evaluate(gat, Z, A_per_type, y, mask=None, label="All Nodes")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'gae_state': gae.state_dict(),
        'gat_state': gat.state_dict(),
        'Z': Z.cpu(),
        'metrics_test': metrics_test,
        'gae_losses': gae_losses,
        'gat_history': gat_history,
    }, results_dir / 'checkpoint.pt')
    print(f"\n[Saved] checkpoint -> {results_dir / 'checkpoint.pt'}")

    return gae, gat, Z, A_per_type, graph, metrics_test


def parse_args():
    p = argparse.ArgumentParser(description="GNN-EADD Phase 1 Training")
    p.add_argument('--n_products',       type=int,   default=200)
    p.add_argument('--n_users',          type=int,   default=80)
    p.add_argument('--n_sellers',        type=int,   default=20)
    p.add_argument('--anomaly_fraction', type=float, default=0.15)
    p.add_argument('--embed_dim',        type=int,   default=128)
    p.add_argument('--hidden_dim',       type=int,   default=64)
    p.add_argument('--gae_epochs',       type=int,   default=200)
    p.add_argument('--gat_epochs',       type=int,   default=100)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--lam',              type=float, default=0.5)
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--results_dir',      type=str,   default='results')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_training(args)
