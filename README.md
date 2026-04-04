# GNN-EADD: E-Commerce Anomaly Detection

This repository implements the Phase 1 pipeline for **GNN-EADD** (Graph Neural Network - E-commerce Anomaly Detection using Dual-stage training).

## Overview

The project detects anomalous behaviour in an e-commerce platform using a dual-stage heterogeneous Graph Neural Network.
1. **Stage 1 (Unsupervised)**: A Graph Auto-Encoder (GAE) learns representations (embeddings) of nodes (Users, Products, Sellers) purely based on graph structure.
2. **Stage 2 (Semi-Supervised)**: A Type-Specific Graph Attention Network (GAT) uses the frozen embeddings to classify anomalies, guided by labeled data and a smoothness penalty (connected nodes share similar scores).

## Structure
- `data/graph_builder.py`: Generates the synthetic heterogeneous graph.
- `models/gae.py`: Stage 1 Unsupervised PyTorch models.
- `models/gat.py`: Stage 2 Semi-supervised PyTorch models.
- `train.py`: Main dual-stage training loop.
- `baseline_comparison.py`: Evaluates against PyTorch Geometric references.
- `visualize.py`: Generates plots (Loss curves, Anomaly distributions, Graph visualisations).
- `run_all.py`: Runs the entire pipeline sequentially.

## Setup & Execution

### 1. Activating the Environment
Before running the code, ensure you have your Python environment activated. If you are using Conda, activate the environment where PyTorch is installed:
```bash
conda activate env_name
```

### 2. Install Dependencies
Make sure all required packages are installed:
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
To run the complete pipeline (Data generation -> GAE -> GAT -> Plotting -> Metric evaluation):
```bash
python run_all.py
```

## Using Your Own Dataset

If you want to train the model manually on a custom dataset rather than the synthetic graph, you need to replace the output of `generate_synthetic_graph()` with your own `HeteroGraph` object.

### Steps to Integrate Custom Data:

1. **Format Your Features:** Prepare your node features as float tensors. You will need separate matrices for each node type (e.g., `x_product`, `x_user`, `x_seller`).
2. **Format Your Edges:** Prepare your connections as edge index tensors (shape `[2, num_edges]`, dtype `torch.long`). For example, a purchase edge index between products and users.
3. **Format Labels & Splits:** Create integer tensors for labels (`0` for normal, `1` for anomalous) and boolean masks representing your Train, Validation, and Test splits.
4. **Build the HeteroGraph Object:** Construct the graph structure using the provided dataclass in `data/graph_builder.py`.
   
   ```python
   from data.graph_builder import HeteroGraph
   import torch

   # 1. Define bounds and counts
   node_counts = {'product': N_P, 'user': N_U, 'seller': N_S}
   bounds = {
       'product': (0, N_P),
       'user': (N_P, N_P + N_U),
       'seller': (N_P + N_U, N_P + N_U + N_S)
   }

   # 2. Package dictionaries
   x_dict = {
       'product': torch.tensor(x_product, dtype=torch.float),
       'user': torch.tensor(x_user, dtype=torch.float),
       'seller': torch.tensor(x_seller, dtype=torch.float),
   }
   
   edge_index_dict = {
       ('product', 'purchase', 'user'): torch.tensor(edge_purchase, dtype=torch.long),
       # Add other relations...
   }

   # 3. Instantiate
   custom_graph = HeteroGraph(
       x_dict=x_dict,
       edge_index_dict=edge_index_dict,
       y=torch.tensor(labels, dtype=torch.long),
       node_type_bounds=bounds,
       num_nodes_per_type=node_counts,
       labeled_mask=torch.tensor(train_mask, dtype=torch.bool),
       val_mask=torch.tensor(val_mask, dtype=torch.bool),
       test_mask=torch.tensor(test_mask, dtype=torch.bool)
   )
   ```
5. **Feed it to `train.py`:** You can then pass this `custom_graph` to the training pipeline in `train.py` instead of the synthetic generator.
