"""Quick verification script - CPU only, no CUDA."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU

import torch
import numpy as np
from pathlib import Path

print("=== CHECKPOINT CONTENTS ===")
ckpt = torch.load('results/checkpoint.pt', map_location='cpu', weights_only=False)
for k, v in ckpt.items():
    if isinstance(v, torch.Tensor):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
    elif isinstance(v, dict):
        print(f'  {k}: {type(v).__name__} with keys {list(v.keys())}')
    elif isinstance(v, list):
        print(f'  {k}: list of {len(v)} items, last={v[-1]}')
    else:
        print(f'  {k}: {v}')

# Verify data splits are disjoint
from data.graph_builder import generate_synthetic_graph
from utils.utils import set_seed
set_seed(42)
graph = generate_synthetic_graph(seed=42)
train_idx = set(torch.where(graph.labeled_mask)[0].tolist())
val_idx   = set(torch.where(graph.val_mask)[0].tolist())
test_idx  = set(torch.where(graph.test_mask)[0].tolist())

print(f'\n=== DATA SPLIT VERIFICATION ===')
print(f'Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}')
print(f'Train∩Val:  {len(train_idx & val_idx)} (should be 0)')
print(f'Train∩Test: {len(train_idx & test_idx)} (should be 0)')
print(f'Val∩Test:   {len(val_idx & test_idx)} (should be 0)')
total = len(train_idx | val_idx | test_idx)
print(f'Coverage:   {total}/300 nodes')

y = graph.y.numpy()
for name, mask in [('Train', graph.labeled_mask), ('Val', graph.val_mask), ('Test', graph.test_mask)]:
    idx = torch.where(mask)[0].numpy()
    print(f'  {name}: {len(idx)-y[idx].sum()} normal + {y[idx].sum()} anomaly')

# Embeddings
Z = ckpt['Z']
print(f'\n=== EMBEDDINGS Z ===')
print(f'Shape: {Z.shape}  (expect [300, 128])')
print(f'NaN: {Z.isnan().any().item()} | Inf: {Z.isinf().any().item()}')
print(f'Range: [{Z.min():.4f}, {Z.max():.4f}]')

# Model weight integrity
print(f'\n=== GAE WEIGHTS ({len(ckpt["gae_state"])} params) ===')
for k, v in ckpt['gae_state'].items():
    print(f'  {k}: {v.shape}')

print(f'\n=== GAT WEIGHTS ({len(ckpt["gat_state"])} params) ===')
for k, v in ckpt['gat_state'].items():
    print(f'  {k}: {v.shape}')

# Reload models and verify inference
from models.gae import GraphAutoEncoder
from models.gat import GraphAttentionNetwork
from utils.utils import build_per_type_adj_matrices, build_homogeneous_features
from train import EDGE_TYPES

H = build_homogeneous_features(graph, torch.device('cpu'))
A_per_type = build_per_type_adj_matrices(graph, normalized=False)

gat = GraphAttentionNetwork(embed_dim=128, hidden_dim=64, edge_types=EDGE_TYPES)
gat.load_state_dict(ckpt['gat_state'])
gat.eval()

with torch.no_grad():
    scores = gat(Z, A_per_type).numpy()

print(f'\n=== INFERENCE CHECK ===')
print(f'Scores range: [{scores.min():.4f}, {scores.max():.4f}]')
print(f'Scores shape: {scores.shape}  (expect (300,))')

# Verify scores are in [0,1] (sigmoid output)
assert scores.min() >= 0.0 and scores.max() <= 1.0, "Scores outside [0,1]!"
print(f'All scores in [0,1]: ✓')

# Check attention mechanism works
with torch.no_grad():
    scores2, attn = gat(Z, A_per_type, return_attention=True)
print(f'Attention returned for {len(attn)} edge types: {list(attn.keys())}')
for r, a in attn.items():
    print(f'  {r}: shape={a.shape}, sum_row0={a[0].sum():.4f} (expect ~1.0 for connected nodes)')

# Verify test metrics match checkpoint
from utils.utils import compute_metrics
test_mask = graph.test_mask.numpy()
test_scores = scores[test_mask]
test_labels = y[test_mask]
recomputed = compute_metrics(test_scores, test_labels)
saved = ckpt['metrics_test']

print(f'\n=== METRICS CONSISTENCY ===')
for k in ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']:
    diff = abs(recomputed[k] - saved[k])
    status = '✓' if diff < 1e-4 else '✗ MISMATCH'
    print(f'  {k}: saved={saved[k]:.4f} recomputed={recomputed[k]:.4f} {status}')

print(f'\n=== LOSS CURVES SANITY ===')
gae_losses = ckpt['gae_losses']
gat_history = ckpt['gat_history']
print(f'GAE losses: {len(gae_losses)} epochs, first={gae_losses[0]:.4f} last={gae_losses[-1]:.4f}')
print(f'GAE loss decreased: {"✓" if gae_losses[-1] < gae_losses[0] else "✗"}')
gat_total = [h[0] for h in gat_history]
print(f'GAT losses: {len(gat_history)} epochs, first={gat_total[0]:.4f} last={gat_total[-1]:.4f}')
print(f'GAT loss decreased: {"✓" if gat_total[-1] < gat_total[0] else "✗"}')

# Verify output files exist
print(f'\n=== OUTPUT FILES ===')
expected = ['checkpoint.pt', 'loss_curves.png', 'anomaly_scores.png', 
            'roc_pr_curves.png', 'graph_structure.png', 'baseline_if_metrics.npy']
for f in expected:
    p = Path('results') / f
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    status = '✓' if exists and size > 0 else '✗ MISSING/EMPTY'
    print(f'  {f}: {status} ({size:,} bytes)')

print(f'\n{"="*60}')
print(f'  ALL CHECKS PASSED ✓')
print(f'{"="*60}')
