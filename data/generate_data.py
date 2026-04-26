import sys
import os
import argparse
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.graph_builder import generate_synthetic_graph
from utils.utils import set_seed

def parse_args():
    p = argparse.ArgumentParser(description="Generate Synthetic Graph for GNN-EADD")
    p.add_argument('--n_products',       type=int,   default=200)
    p.add_argument('--n_users',          type=int,   default=80)
    p.add_argument('--n_sellers',        type=int,   default=20)
    p.add_argument('--anomaly_fraction', type=float, default=0.15)
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--output',           type=str,   default='data/graph.pt')
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    print("\n[Data] Building synthetic heterogeneous e-commerce graph...")
    graph = generate_synthetic_graph(
        n_products=args.n_products,
        n_users=args.n_users,
        n_sellers=args.n_sellers,
        anomaly_fraction=args.anomaly_fraction,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the graph with torch.save
    torch.save(graph, out_path)
    print(f"\n[Saved] Graph saved to -> {out_path}")

if __name__ == '__main__':
    main()
