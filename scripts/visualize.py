"""
visualize.py  —  Plots for GNN-EADD Phase 1

Generates:
  1. Loss curves for GAE and GAT stages
  2. Anomaly score distribution (normal vs anomalous nodes)
  3. Graph structure visualization (colour-coded by type and anomaly status)
  4. Attention weight heatmap for a sample node
  5. ROC and PR curves

Why do programmers prefer dark mode? Because light attracts bugs.
"""

import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.graph_builder import generate_synthetic_graph
from models.gae import GraphAutoEncoder
from models.gat import GraphAttentionNetwork
from utils.utils import (
    set_seed, build_per_type_adj_matrices,
    build_homogeneous_features,
)
from train import EDGE_TYPES


SAVE_DIR = Path('results')
SAVE_DIR.mkdir(exist_ok=True)

# ── PALETTE ──────────────────────────────────────────────────────────────────
C_PRODUCT  = '#4C72B0'   # blue
C_USER     = '#55A868'   # green
C_SELLER   = '#C44E52'   # red-ish (normal seller)
C_ANOMALY  = '#DD8452'   # orange — anomalous node highlight
C_NORMAL   = '#6c757d'   # grey


def plot_loss_curves(gae_losses: list, gat_history: list, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # GAE loss
    axes[0].plot(gae_losses, color='#4C72B0', linewidth=1.5)
    axes[0].set_title('Stage 1 — GAE Reconstruction Loss', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].grid(True, alpha=0.3)

    # GAT losses
    epochs = range(1, len(gat_history) + 1)
    total  = [h[0] for h in gat_history]
    sup    = [h[1] for h in gat_history]
    unsup  = [h[2] for h in gat_history]
    axes[1].plot(epochs, total, label='Total', color='#4C72B0', linewidth=1.8)
    axes[1].plot(epochs, sup,   label='L_sup', color='#C44E52', linewidth=1.2, linestyle='--')
    axes[1].plot(epochs, unsup, label='L_unsup', color='#55A868', linewidth=1.2, linestyle=':')
    axes[1].set_title('Stage 2 — GAT Loss (L_sup + λ L_unsup)', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = SAVE_DIR / 'loss_curves.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {path}")
    plt.close()


def plot_anomaly_scores(scores: np.ndarray, labels: np.ndarray, node_types: np.ndarray, save: bool = True):
    """
    Histogram of anomaly scores split by normal / anomalous.
    Also shows per-node-type breakdown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Overall distribution
    normal_scores  = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    bins = np.linspace(0, 1, 30)
    axes[0].hist(normal_scores,  bins=bins, alpha=0.7, label='Normal',   color=C_NORMAL)
    axes[0].hist(anomaly_scores, bins=bins, alpha=0.7, label='Anomalous', color=C_ANOMALY)
    axes[0].axvline(0.5, color='black', linestyle='--', linewidth=1.2, label='Threshold=0.5')
    axes[0].set_title('Anomaly Score Distribution', fontsize=12)
    axes[0].set_xlabel('Anomaly Score s_i')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-node-type breakdown
    type_names  = ['Product', 'User', 'Seller']
    type_colors = [C_PRODUCT, C_USER, C_SELLER]
    x_pos = np.arange(len(type_names))
    means = [scores[node_types == t].mean() if (node_types == t).any() else 0.0
             for t in range(3)]
    stds  = [scores[node_types == t].std()  if (node_types == t).any() else 0.0
             for t in range(3)]
    axes[1].bar(x_pos, means, yerr=stds, color=type_colors, alpha=0.8, capsize=5)
    axes[1].axhline(0.5, color='black', linestyle='--', linewidth=1.2)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(type_names)
    axes[1].set_title('Mean Anomaly Score by Node Type', fontsize=12)
    axes[1].set_ylabel('Mean Score')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        path = SAVE_DIR / 'anomaly_scores.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {path}")
    plt.close()


def plot_roc_pr(scores: np.ndarray, labels: np.ndarray, save: bool = True):
    """ROC curve and Precision-Recall curve."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='#4C72B0', linewidth=2, label=f'AUC-ROC = {roc_auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_title('ROC Curve', fontsize=12)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PR
    prec, rec, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(rec, prec)
    axes[1].plot(rec, prec, color='#C44E52', linewidth=2, label=f'AUC-PR = {pr_auc:.4f}')
    axes[1].set_title('Precision-Recall Curve', fontsize=12)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = SAVE_DIR / 'roc_pr_curves.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {path}")
    plt.close()


def plot_graph_structure(graph, scores: np.ndarray, labels: np.ndarray, save: bool = True):
    """
    Spring-layout visualization of the graph with:
      - Node colour = type (product/user/seller)
      - Node border  = red if anomalous, grey if normal
      - Node size    = proportional to anomaly score
    """
    try:
        import networkx as nx
    except ImportError:
        print("[Plot] networkx not installed, skipping graph visualization.")
        return

    N_P = graph.num_nodes_per_type['product']
    N_U = graph.num_nodes_per_type['user']
    N_S = graph.num_nodes_per_type['seller']
    N   = N_P + N_U + N_S

    if N > 2000:
        print(f"[Plot] Graph too large for networkx spring_layout ({N} nodes). Skipping structure visualization.")
        return

    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Add edges
    ei = graph.edge_index_dict.get(('product', 'purchase', 'user'),
                                   torch.zeros(2, 0, dtype=torch.long))
    for p, u in zip(ei[0].tolist(), ei[1].tolist()):
        G.add_edge(p, N_P + u, etype='purchase')

    ei = graph.edge_index_dict.get(('seller', 'sell', 'product'),
                                   torch.zeros(2, 0, dtype=torch.long))
    for s, p in zip(ei[0].tolist(), ei[1].tolist()):
        G.add_edge(N_P + N_U + s, p, etype='sell')

    ei = graph.edge_index_dict.get(('user', 'interact', 'user'),
                                   torch.zeros(2, 0, dtype=torch.long))
    for u1, u2 in zip(ei[0].tolist(), ei[1].tolist()):
        G.add_edge(N_P + u1, N_P + u2, etype='interact')

    node_colors = (
        [C_PRODUCT] * N_P + [C_USER] * N_U + [C_SELLER] * N_S
    )
    border_colors = ['#A32D2D' if labels[i] == 1 else '#aaaaaa' for i in range(N)]
    node_sizes    = [200 + 800 * float(scores[i]) for i in range(N)]

    pos = nx.spring_layout(G, seed=42, k=2.5)

    fig, ax = plt.subplots(figsize=(14, 9))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           edgecolors=border_colors, linewidths=1.8,
                           node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#888888', ax=ax)

    # Labels (only for anomalous nodes)
    anom_labels = {i: f"{'P' if i<N_P else 'U' if i<N_P+N_U else 'S'}{i}"
                   for i in range(N) if labels[i] == 1}
    nx.draw_networkx_labels(G, pos, labels=anom_labels, font_size=7,
                            font_color='black', ax=ax)

    legend_handles = [
        mpatches.Patch(color=C_PRODUCT, label='Product'),
        mpatches.Patch(color=C_USER,    label='User'),
        mpatches.Patch(color=C_SELLER,  label='Seller'),
        mpatches.Patch(facecolor='white', edgecolor='#A32D2D',
                       linewidth=2, label='Anomalous (red border)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9)
    ax.set_title('E-Commerce Heterogeneous Graph\n(node size = anomaly score)', fontsize=13)
    ax.axis('off')
    plt.tight_layout()

    if save:
        path = SAVE_DIR / 'graph_structure.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {path}")
    plt.close()


def make_all_plots(
    gae_losses, gat_history, scores_np, labels_np, node_types_np, graph
):
    print("\n[Visualization] Generating plots...")
    plot_loss_curves(gae_losses, gat_history)
    plot_anomaly_scores(scores_np, labels_np, node_types_np)
    if labels_np.sum() > 0 and labels_np.sum() < len(labels_np):
        plot_roc_pr(scores_np, labels_np)
    plot_graph_structure(graph, scores_np, labels_np)
    print(f"[Visualization] All plots saved to {SAVE_DIR}/")


if __name__ == '__main__':
    # Load saved checkpoint and regenerate plots
    ckpt_path = SAVE_DIR / 'checkpoint.pt'
    if not ckpt_path.exists():
        print(f"[Error] Run train.py first to generate {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"[Loaded] checkpoint from {ckpt_path}")
    print(f"  Metrics (all nodes): {ckpt['metrics_all']}")

    # Re-build graph for visualization
    set_seed(42)
    graph = generate_synthetic_graph()
    N_P = graph.num_nodes_per_type['product']
    N_U = graph.num_nodes_per_type['user']
    N_S = graph.num_nodes_per_type['seller']
    N   = N_P + N_U + N_S
    node_types_np = np.array([0]*N_P + [1]*N_U + [2]*N_S)

    device = torch.device('cpu')
    H = build_homogeneous_features(graph, device)
    A_per_type = build_per_type_adj_matrices(graph, normalized=False)
    feat_dim = H.shape[1]

    gae = GraphAutoEncoder(feat_dim=feat_dim, hidden_dim=64, embed_dim=128, edge_types=EDGE_TYPES)
    gae.load_state_dict(ckpt['gae_state'])
    gat = GraphAttentionNetwork(embed_dim=128, hidden_dim=64, edge_types=EDGE_TYPES)
    gat.load_state_dict(ckpt['gat_state'])

    Z = ckpt['Z']
    gat.eval()
    with torch.no_grad():
        scores  = gat(Z, A_per_type).numpy()
    labels = graph.y.numpy()

    make_all_plots(
        ckpt['gae_losses'], ckpt['gat_history'],
        scores, labels, node_types_np, graph
    )
