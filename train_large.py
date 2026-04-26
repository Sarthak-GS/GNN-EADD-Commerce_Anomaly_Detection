import sys
import os
import torch
import torch.optim as optim
import time
import argparse
from pathlib import Path
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gae import GraphAutoEncoder
from models.gat import GraphAttentionNetwork
from utils.utils import set_seed, build_homogeneous_features, compute_metrics
from train import EDGE_TYPES

def train_minibatch(args):
    """
    Mini-batch training for graphs so big they make my CPU sweat :(
    """
    set_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using: {dev}")

    print(f"\n[Data] Loading {args.graph_path}...")
    graph = torch.load(args.graph_path, weights_only=False)
    H_cpu = build_homogeneous_features(graph, torch.device('cpu'))
    
    N_P, N_U, N_S = graph.num_nodes_per_type['product'], graph.num_nodes_per_type['user'], graph.num_nodes_per_type['seller']
    
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

    g_edge_index = torch.cat(parts, dim=1) if parts else torch.zeros(2, 0, dtype=torch.long)
    print(f"[Graph] Loaded {H_cpu.size(0)} nodes and {g_edge_index.size(1)} edges.")

    pyg_data = Data(x=H_cpu, edge_index=g_edge_index)
    loader = NeighborLoader(pyg_data, num_neighbors=[5, 5], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    gae = GraphAutoEncoder(H_cpu.shape[1], args.hidden_dim, args.embed_dim, EDGE_TYPES).to(dev)
    gat = GraphAttentionNetwork(args.embed_dim, args.hidden_dim, EDGE_TYPES).to(dev)
    gae_opt = optim.Adam(gae.parameters(), lr=args.lr)
    gat_opt = optim.Adam(gat.parameters(), lr=args.lr)
    
    def get_batch_adj(ei, size):
        A = torch.zeros(size, size, device=dev)
        A.fill_diagonal_(1.0)
        if ei.numel() > 0: A[ei[0], ei[1]] = 1.0
        return A
        
    def norm_adj(A):
        d = A.sum(dim=1).pow(-0.5)
        d[d == float('inf')] = 0.0
        D = torch.diag(d)
        return D @ A @ D

    print("\n" + "="*60 + "\n  MINI-BATCH TRAINING FOR MASSIVE GRAPHS\n" + "="*60)
    
    gae.train()
    for e in range(1, args.gae_epochs + 1):
        loss_acc = 0
        for b in loader:
            b = b.to(dev)
            A_b = get_batch_adj(b.edge_index, b.num_nodes)
            A_n = {r: norm_adj(A_b) for r in EDGE_TYPES}
            gae_opt.zero_grad()
            _, A_hat = gae(b.x, A_n)
            loss = gae.reconstruction_loss(A_b, A_hat)
            loss.backward(); gae_opt.step()
            loss_acc += loss.item()
        if e % 5 == 0 or e == 1: print(f"  [GAE E{e:>3}]  loss = {loss_acc/len(loader):.5f}")

    gat.train(); gae.eval()
    gy, gm = graph.y.cpu(), graph.labeled_mask.cpu()
    for e in range(1, args.gat_epochs + 1):
        loss_acc = 0
        for b in loader:
            b = b.to(dev)
            A_b = get_batch_adj(b.edge_index, b.num_nodes)
            A_n = {r: norm_adj(A_b) for r in EDGE_TYPES}
            with torch.no_grad(): Z_b = gae.encode(b.x, A_n).detach()
            gat_opt.zero_grad()
            s = gat(Z_b, {r: A_b for r in EDGE_TYPES})
            loss, _, _ = GraphAttentionNetwork.combined_loss(s, gy[b.n_id].to(dev), gm[b.n_id].to(dev), A_b, args.lam)
            loss.backward(); gat_opt.step()
            loss_acc += loss.item()
        if e % 5 == 0 or e == 1: print(f"  [GAT E{e:>3}]  loss = {loss_acc/len(loader):.5f}")
            
    print("\n[✓] Mini-batch training complete!")

    # ── EVALUATION ───────────────────────────────────────────────────
    print("\n" + "="*60 + "\n  STAGE 3: Evaluation (on test split)\n" + "="*60)
    gat.eval(); gae.eval()
    all_s, all_y = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(dev)
            A_b = get_batch_adj(b.edge_index, b.num_nodes)
            A_n = {r: norm_adj(A_b) for r in EDGE_TYPES}
            Z_b = gae.encode(b.x, A_n)
            s = gat(Z_b, {r: A_b for r in EDGE_TYPES})
            mask = graph.test_mask[b.n_id].to(dev)
            if mask.any():
                all_s.append(s[mask].cpu())
                all_y.append(gy[b.n_id][mask].cpu())

    if all_s:
        all_s, all_y = torch.cat(all_s).numpy(), torch.cat(all_y).numpy()
        m = compute_metrics(all_s, all_y)
        for k, v in m.items(): print(f"  {k:<10}: {v:.4f}")
    else:
        print(" [!] No test nodes found.")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--graph_path', default='data/graph.pt')
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--gae_epochs', type=int, default=20)
    p.add_argument('--gat_epochs', type=int, default=20)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lam', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=0)
    train_minibatch(p.parse_args())
