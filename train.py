import sys
import os
import torch
import torch.optim as optim
import argparse
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.graph_builder import build_global_adjacency
from models.gae import GraphAutoEncoder
from models.gat import GraphAttentionNetwork
from utils.utils import (
    set_seed,
    build_per_type_adj_matrices,
    build_homogeneous_features,
    compute_metrics,
    move_adj_dict_to_device,
)

EDGE_TYPES = ['purchase', 'sell', 'interact']


def select_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")
    return device


def train_gae(gae, H, A, A_norm, n_epochs=200, lr=1e-3, log_every=20):
    """
    Stage 1: Training the GAE to find itself :(
    """
    optimizer = optim.Adam(gae.parameters(), lr=lr)
    losses = []
    print("\n" + "="*60 + "\n  STAGE 1: GAE Unsupervised Training\n" + "="*60)
    
    start = time.time()
    gae.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        Z, A_hat = gae(H, A_norm)
        loss = gae.reconstruction_loss(A, A_hat)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % log_every == 0 or epoch == 1:
            print(f"  [GAE E{epoch:>4}]  loss = {loss.item():.6f}")
    
    dt = time.time() - start
    print(f"  GAE done in {dt:.3f}s. Final loss: {losses[-1]:.6f}")
    return losses, dt


def train_gat(gat, Z, A, A_rel, y, mask, n_epochs=100, lr=1e-3, lam=0.5, log_every=10):
    """
    Stage 2: Training the GAT to find the anomalies :)
    """
    optimizer = optim.Adam(gat.parameters(), lr=lr)
    history = []
    print("\n" + "="*60 + "\n  STAGE 2: GAT Semi-supervised Training\n" + "="*60)
    
    start = time.time()
    gat.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        s = gat(Z, A_rel)
        loss, sup, unsup = GraphAttentionNetwork.combined_loss(s, y, mask, A, lam)
        loss.backward()
        optimizer.step()
        history.append((loss.item(), sup.item(), unsup.item()))
        if epoch % log_every == 0 or epoch == 1:
            print(f"  [GAT E{epoch:>4}]  loss={loss.item():.5f} sup={sup.item():.5f} unsup={unsup.item():.5f}")
            
    dt = time.time() - start
    print(f"  GAT done in {dt:.3f}s. Final loss: {history[-1][0]:.5f}")
    return history, dt


@torch.no_grad()
def evaluate(gat, Z, A_rel, y, mask=None, label="Test"):
    """ Evaluating performance, hopefully we are good :) """
    gat.eval()
    scores = gat(Z, A_rel).cpu().numpy()
    labels = y.cpu().numpy()
    if mask is not None:
        m = mask.cpu().numpy()
        scores, labels = scores[m], labels[m]
    
    m = compute_metrics(scores, labels)
    print(f"\n[{label} Results]")
    for k, v in m.items():
        print(f"  {k:<10}: {v:.4f}")
    return m


def run_training(args):
    set_seed(args.seed)
    dev = select_device()
    
    print(f"\n[Data] Loading {args.graph_path}...")
    graph = torch.load(args.graph_path, weights_only=False)
    
    H = build_homogeneous_features(graph, dev)
    A = build_global_adjacency(graph).to(dev)
    A_norm = move_adj_dict_to_device(build_per_type_adj_matrices(graph, True), dev)
    A_rel = move_adj_dict_to_device(build_per_type_adj_matrices(graph, False), dev)
    
    y, mask = graph.y.to(dev), graph.labeled_mask.to(dev)
    v_mask = getattr(graph, 'val_mask', torch.zeros_like(mask)).to(dev)
    t_mask = graph.test_mask.to(dev)

    gae = GraphAutoEncoder(H.shape[1], args.hidden_dim, args.embed_dim, EDGE_TYPES).to(dev)
    gae_l, gae_t = train_gae(gae, H, A, A_norm, args.gae_epochs, args.lr, max(1, args.gae_epochs // 10))
    
    gae.eval()
    with torch.no_grad():
        Z = gae.encode(H, A_norm).detach()

    gat = GraphAttentionNetwork(args.embed_dim, args.hidden_dim, EDGE_TYPES).to(dev)
    gat_h, gat_t = train_gat(gat, Z, A, A_rel, y, mask, args.gat_epochs, args.lr, args.lam, max(1, args.gat_epochs // 10))

    if v_mask.any(): evaluate(gat, Z, A_rel, y, v_mask, "Validation")
    m_test = evaluate(gat, Z, A_rel, y, t_mask if t_mask.any() else None, "Test")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        'gae': gae.state_dict(),
        'gat': gat.state_dict(),
        'gae_state': gae.state_dict(),
        'gat_state': gat.state_dict(),
        'Z': Z.cpu(),
        'metrics': m_test,
        'gae_losses': gae_l,
        'gat_history': gat_h,
        'times': {'gae': gae_t, 'gat': gat_t},
    }, Path(args.results_dir) / 'checkpoint.pt')
    
    m_test['gae_time'], m_test['gat_time'] = gae_t, gat_t
    return gae, gat, Z, A_rel, graph, m_test, H, A_norm


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--graph_path', default='data/graph.pt')
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--gae_epochs', type=int, default=200)
    p.add_argument('--gat_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lam', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--results_dir', default='results')
    run_training(p.parse_args())
