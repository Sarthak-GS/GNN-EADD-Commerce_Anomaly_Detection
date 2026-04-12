"""
run_all.py  —  Master script that runs the complete GNN-EADD Phase 1 pipeline:
  1. Builds synthetic graph
  2. Trains GAE (Stage 1)
  3. Trains GAT (Stage 2)
  4. Evaluates metrics
  5. Generates all plots
  6. Runs baseline comparisons
  7. Prints final Phase 1 summary

Usage:
    python generate_data.py --output data/graph.pt
    python run_all.py
    python run_all.py --gae_epochs 50 --gat_epochs 30  # fast demo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from types import SimpleNamespace
from pathlib import Path

from train import run_training
from visualize import make_all_plots
from baseline_comparison import run_baseline_comparison
from utils.utils import set_seed, build_per_type_adj_matrices, build_homogeneous_features


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gae_epochs',       type=int,   default=200)
    p.add_argument('--gat_epochs',       type=int,   default=100)
    p.add_argument('--embed_dim',        type=int,   default=128)
    p.add_argument('--hidden_dim',       type=int,   default=64)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--lam',              type=float, default=0.5)
    p.add_argument('--graph_path',       type=str,   default='data/graph.pt')
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--results_dir',      type=str,   default='results')
    p.add_argument('--skip_baselines',   action='store_true')
    p.add_argument('--decoder_type',     type=str,   default='inner_product',
                   choices=['inner_product', 'mlp'])
    p.add_argument('--parallel_mode',    type=str,   default='sequential',
                   choices=['sequential', 'openmp', 'cuda'])
    p.add_argument('--n_threads',        type=int,   default=4)
    args = p.parse_args()

    # ── TRAIN ────────────────────────────────────────────────────────────────
    gae, gat, Z, A_per_type, graph, metrics_all = run_training(args)

    # ── VISUALIZE ────────────────────────────────────────────────────────────
    ckpt = torch.load(Path(args.results_dir) / 'checkpoint.pt', map_location='cpu', weights_only=False)

    set_seed(args.seed)
    N_P = graph.num_nodes_per_type['product']
    N_U = graph.num_nodes_per_type['user']
    N_S = graph.num_nodes_per_type['seller']
    N   = N_P + N_U + N_S
    node_types_np = np.array([0]*N_P + [1]*N_U + [2]*N_S)

    gat.eval()
    with torch.no_grad():
        scores_t = gat(Z, A_per_type)
    scores_np = scores_t.cpu().numpy()
    labels_np = graph.y.numpy()

    make_all_plots(
        gae_losses   = ckpt['gae_losses'],
        gat_history  = ckpt['gat_history'],
        scores_np    = scores_np,
        labels_np    = labels_np,
        node_types_np= node_types_np,
        graph        = graph,
    )

    # ── BASELINES ────────────────────────────────────────────────────────────
    if not args.skip_baselines:
        run_baseline_comparison(args.graph_path)

    # ── PHASE 1 SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "█"*64)
    print("  GNN-EADD PHASE 2 — FINAL SUMMARY")
    print("█"*64)
    print(f"  Decoder   : {args.decoder_type}")
    print(f"  Parallel  : {args.parallel_mode}")
    print(f"\n  Graph:  {N_P} products | {N_U} users | {N_S} sellers | {N} total nodes")
    print(f"  Labels: {labels_np.sum()} anomalies ({100*labels_np.mean():.1f}% of nodes)")
    print()
    print(f"  ┌─────────────────────┬──────────┐")
    print(f"  │ Metric              │  Score   │")
    print(f"  ├─────────────────────┼──────────┤")
    print(f"  │ AUC-ROC             │  {metrics_all['auc_roc']:.4f}  │")
    print(f"  │ AUC-PR              │  {metrics_all['auc_pr']:.4f}  │")
    print(f"  │ F1-score            │  {metrics_all['f1']:.4f}  │")
    print(f"  │ Precision           │  {metrics_all['precision']:.4f}  │")
    print(f"  │ Recall              │  {metrics_all['recall']:.4f}  │")
    print(f"  │ Precision@K         │  {metrics_all['prec_at_k']:.4f}  │")
    print(f"  │ Recall@K            │  {metrics_all['recall_at_k']:.4f}  │")
    print(f"  └─────────────────────┴──────────┘")
    baseline_path = Path(args.results_dir) / 'baseline_pyg_metrics.npy'
    if not args.skip_baselines and baseline_path.exists():
        metrics_pyg = np.load(baseline_path, allow_pickle=True).item()
        print(f"  ┌─────────────────────┬───────────┬───────────┐")
        print(f"  │ Runtime Comparison  │ Ours (s)  │ PyG (s)   │")
        print(f"  ├─────────────────────┼───────────┼───────────┤")
        print(f"  │ Stage 1 (GAE)       │ {metrics_all.get('gae_time', 0.0):>9.3f} │ {metrics_pyg.get('gae_time', 0.0):>9.3f} │")
        print(f"  │ Stage 2 (GAT)       │ {metrics_all.get('gat_time', 0.0):>9.3f} │ {metrics_pyg.get('gat_time', 0.0):>9.3f} │")
        print(f"  ├─────────────────────┼───────────┼───────────┤")
        print(f"  │ Total Time          │ {(metrics_all.get('gae_time', 0.0) + metrics_all.get('gat_time', 0.0)):>9.3f} │ {(metrics_pyg.get('gae_time', 0.0) + metrics_pyg.get('gat_time', 0.0)):>9.3f} │")
        print(f"  └─────────────────────┴───────────┴───────────┘")
        print()

    print(f"  Plots saved to: {args.results_dir}/")
    print(f"    • loss_curves.png")
    print(f"    • anomaly_scores.png")
    print(f"    • roc_pr_curves.png")
    print(f"    • graph_structure.png  (requires networkx)")
    print()
    print(f"  Checkpoint saved to: {args.results_dir}/checkpoint.pt")
    print("█"*64 + "\n")


if __name__ == '__main__':
    main()
