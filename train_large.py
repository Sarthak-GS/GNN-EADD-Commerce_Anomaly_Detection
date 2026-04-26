"""
train_large.py — GNN-EADD Training Pipeline for Massive Graphs (Millions of Nodes)

This script uses PyTorch Geometric's `NeighborLoader` to perform GraphSAGE-style
mini-batching. This allows us to train the PyTorch Dense baseline on graphs with
millions of nodes without running out of VRAM!

It works by sampling subgraphs (e.g., 2000 nodes at a time), creating small dense
matrices for those subgraphs, and training the model on them.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.optim as optim
import time
from pathlib import Path

# We import PyG specifically for the NeighborLoader sampler
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

from models.gae import GraphAutoEncoder
from models.gat import GraphAttentionNetwork
from utils.utils import set_seed, build_homogeneous_features
from train import EDGE_TYPES

def train_minibatch(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {device}")

    # 1. Load Graph
    print(f"\n[Data] Loading graph from {args.graph_path}...")
    graph = torch.load(args.graph_path)
    N = sum(graph.num_nodes_per_type.values())
    
    # 2. Build Global Node Features
    H_global = build_homogeneous_features(graph, torch.device('cpu')) # keep on CPU for massive graphs
    
    # 3. Build Global Sparse Edge Index
    # We must extract the edge_index manually to avoid creating an N x N dense matrix!
    N_P = graph.num_nodes_per_type['product']
    N_U = graph.num_nodes_per_type['user']
    N_S = graph.num_nodes_per_type['seller']
    
    parts = []
    # Purchase
    ei = graph.edge_index_dict.get(('product', 'purchase', 'user'), torch.zeros(2, 0, dtype=torch.long))
    if ei.numel() > 0:
        parts.append(torch.stack([ei[0], N_P + ei[1]], dim=0))
        parts.append(torch.stack([N_P + ei[1], ei[0]], dim=0))
    # Sell
    ei = graph.edge_index_dict.get(('seller', 'sell', 'product'), torch.zeros(2, 0, dtype=torch.long))
    if ei.numel() > 0:
        parts.append(torch.stack([N_P + N_U + ei[0], ei[1]], dim=0))
        parts.append(torch.stack([ei[1], N_P + N_U + ei[0]], dim=0))
    # Interact
    ei = graph.edge_index_dict.get(('user', 'interact', 'user'), torch.zeros(2, 0, dtype=torch.long))
    if ei.numel() > 0:
        parts.append(torch.stack([N_P + ei[0], N_P + ei[1]], dim=0))
        parts.append(torch.stack([N_P + ei[1], N_P + ei[0]], dim=0))

    if len(parts) > 0:
        global_edge_index = torch.cat(parts, dim=1)
    else:
        global_edge_index = torch.zeros(2, 0, dtype=torch.long)
        
    print(f"[Graph] Loaded {N} nodes and {global_edge_index.size(1)} edges.")

    # 4. Create PyG Data object for the sampler
    pyg_data = Data(x=H_global, edge_index=global_edge_index)
    
    # Initialize the mini-batch sampler!
    # This samples 2 hops of neighbors: 10 neighbors in hop 1, 10 in hop 2.
    loader = NeighborLoader(
        pyg_data,
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    # Initialize Models
    gae = GraphAutoEncoder(H_global.shape[1], args.hidden_dim, args.embed_dim, EDGE_TYPES).to(device)
    gat = GraphAttentionNetwork(args.embed_dim, args.hidden_dim, EDGE_TYPES).to(device)
    
    gae_opt = optim.Adam(gae.parameters(), lr=args.lr)
    gat_opt = optim.Adam(gat.parameters(), lr=args.lr)
    
    # Helper to build dense matrices FOR THE BATCH ONLY
    def build_batch_adj(batch_edge_index, batch_size):
        A = torch.zeros(batch_size, batch_size, device=device)
        A.fill_diagonal_(1.0)
        if batch_edge_index.numel() > 0:
            A[batch_edge_index[0], batch_edge_index[1]] = 1.0
        return A
        
    def normalize_batch_adj(A):
        deg = A.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        D = torch.diag(deg_inv_sqrt)
        return D @ A @ D

    print(f"\n{'='*60}")
    print("  MINI-BATCH TRAINING FOR MASSIVE GRAPHS")
    print(f"{'='*60}")
    
    # ── STAGE 1: GAE ──────────────────────────────────────────────────────────
    gae.train()
    for epoch in range(1, args.gae_epochs + 1):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            B = batch.num_nodes
            
            # Reconstruct small dense matrices for this subgraph
            A_batch = build_batch_adj(batch.edge_index, B)
            A_norm = normalize_batch_adj(A_batch)
            A_norm_per_type = {r: A_norm for r in EDGE_TYPES} # simplified for batch
            
            gae_opt.zero_grad()
            Z, A_hat = gae(batch.x, A_norm_per_type)
            loss = gae.reconstruction_loss(A_batch, A_hat)
            loss.backward()
            gae_opt.step()
            
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [GAE E{epoch:>3}]  loss = {total_loss / len(loader):.5f}")

    # ── STAGE 2: GAT ──────────────────────────────────────────────────────────
    # To train GAT, we need labels.
    global_y = graph.y.to('cpu')
    global_mask = graph.labeled_mask.to('cpu')
    
    gat.train()
    gae.eval()
    for epoch in range(1, args.gat_epochs + 1):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            B = batch.num_nodes
            
            # Build matrices
            A_batch = build_batch_adj(batch.edge_index, B)
            A_norm = normalize_batch_adj(A_batch)
            A_norm_per_type = {r: A_norm for r in EDGE_TYPES}
            A_per_type = {r: A_batch for r in EDGE_TYPES}
            
            with torch.no_grad():
                Z_batch = gae.encode(batch.x, A_norm_per_type).detach()
                
            gat_opt.zero_grad()
            s = gat(Z_batch, A_per_type)
            
            batch_y = global_y[batch.n_id].to(device)
            batch_mask = global_mask[batch.n_id].to(device)
            
            loss, _, _ = GraphAttentionNetwork.combined_loss(
                s, batch_y, batch_mask, A_batch, lam=args.lam
            )
            
            loss.backward()
            gat_opt.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [GAT E{epoch:>3}]  loss = {total_loss / len(loader):.5f}")
            
    print("\n[✓] Mini-batch training complete!")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--graph_path', type=str, default='data/graph.pt')
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--gae_epochs', type=int, default=20)
    p.add_argument('--gat_epochs', type=int, default=20)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lam', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    
    train_minibatch(args)
