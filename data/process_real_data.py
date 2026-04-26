import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.graph_builder import HeteroGraph

def create_real_graph():
    """
    Turning messy Amazon reviews into a clean graph.
    It's like tidying up your room, but with data :)
    """
    input_path = Path("real-data/All_Beauty.jsonl")
    if not input_path.exists():
        print(f"[Error] {input_path} not found!")
        return

    users, products, edges = {}, {}, []

    print("[Data] Reading All_Beauty.jsonl...")
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 100000: break
            d = json.loads(line)
            uid, pid = d['user_id'], d['parent_asin']
            if uid not in users: users[uid] = len(users)
            if pid not in products: products[pid] = len(products)
            edges.append((products[pid], users[uid]))

    np_p, np_u, np_s = len(products), len(users), 1
    print(f"[Graph] Nodes: {np_p} products, {np_u} users, 1 seller")
    
    # Random features since we don't have text embeddings yet :(
    xp = np.random.uniform(0.2, 0.8, (np_p, 8)).astype(np.float32)
    xu = np.random.uniform(0.2, 0.8, (np_u, 6)).astype(np.float32)
    xs = np.random.uniform(0.2, 0.8, (np_s, 5)).astype(np.float32)

    sell_edges = torch.stack([torch.zeros(np_p, dtype=torch.long), torch.arange(np_p, dtype=torch.long)], dim=0)
    purchase_edges = torch.tensor(edges, dtype=torch.long).t()

    y = torch.zeros(np_p + np_u + np_s, dtype=torch.long)
    anom_p = np.random.choice(np_p, int(0.05 * np_p), replace=False)
    anom_u = np.random.choice(np_u, int(0.05 * np_u), replace=False)
    y[anom_p] = 1
    y[np_p + anom_u] = 1

    idx = np.random.permutation(len(y))
    m_train, m_val, m_test = torch.zeros_like(y, dtype=torch.bool), torch.zeros_like(y, dtype=torch.bool), torch.zeros_like(y, dtype=torch.bool)
    m_train[idx[:int(0.6*len(y))]] = True
    m_val[idx[int(0.6*len(y)):int(0.8*len(y))]] = True
    m_test[idx[int(0.8*len(y)):]] = True

    graph = HeteroGraph(
        x_dict={'product': torch.from_numpy(xp), 'user': torch.from_numpy(xu), 'seller': torch.from_numpy(xs)},
        edge_index_dict={('product', 'purchase', 'user'): purchase_edges, ('seller', 'sell', 'product'): sell_edges, ('user', 'interact', 'user'): torch.zeros(2, 0, dtype=torch.long)},
        y=y,
        node_type_bounds={'product': (0, np_p), 'user': (np_p, np_p + np_u), 'seller': (np_p+np_u, np_p+np_u+np_s)},
        num_nodes_per_type={'product': np_p, 'user': np_u, 'seller': np_s},
        labeled_mask=m_train, val_mask=m_val, test_mask=m_test
    )
    torch.save(graph, "data/real_graph.pt")
    print("[✓] Saved to data/real_graph.pt")

if __name__ == "__main__":
    create_real_graph()
