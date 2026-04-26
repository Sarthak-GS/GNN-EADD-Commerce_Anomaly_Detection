# GNN-EADD: E-Commerce Anomaly Detection

This repository implements the Phase 1 pipeline for **GNN-EADD** (Graph Neural Network - E-commerce Anomaly Detection using Dual-stage training).

## Overview

The project detects anomalous behaviour in an e-commerce platform using a dual-stage heterogeneous Graph Neural Network.
1. **Stage 1 (Unsupervised)**: A Graph Auto-Encoder (GAE) learns representations (embeddings) of nodes (Users, Products, Sellers) purely based on graph structure.
2. **Stage 2 (Semi-Supervised)**: A Type-Specific Graph Attention Network (GAT) uses the frozen embeddings to classify anomalies, guided by labeled data and a smoothness penalty (connected nodes share similar scores).

## Structure
- `generate_data.py`: Creates the synthetic graph and saves it to a file.
- `data/graph_builder.py`: Generates the synthetic heterogeneous graph.
- `models/gae.py`: Stage 1 Unsupervised PyTorch models.
- `models/gat.py`: Stage 2 Semi-supervised PyTorch models.
- `train_large.py`: Main dual-stage training loop (with PyG Mini-Batching for massive graphs).
- `baseline_comparison.py`: Evaluates against PyTorch Geometric references.
- `visualize.py`: Generates plots (Loss curves, Anomaly distributions, Graph visualisations).
- `run_phase2.py`: Master script to run training, benchmarks, and custom CUDA inference.

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
The graph generation is now decoupled from model training. First, generate the synthetic graph data:
```bash
python generate_data.py --output data/graph.pt
```

Then, to run the complete pipeline (GAE -> GAT -> Inference Benchmarks -> Plotting):
```bash
python run_phase2.py --graph_path data/graph.pt
```

## Using Your Own Dataset

If you want to train the model manually on a custom dataset rather than the synthetic graph, you need to save your own `HeteroGraph` object as a `.pt` file and pass it to the training scripts.

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

   # 3. Instantiate and save
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
   
   # Save the graph
   torch.save(custom_graph, "data/my_custom_graph.pt")
   ```
5. **Feed it to the pipeline:** You can then pass this saved file to the training pipeline using the `--graph_path` argument:
   ```bash
   python run_phase2.py --graph_path data/my_custom_graph.pt
   ```

## Command-Line Arguments

The pipeline is highly customizable. You can modify the architecture and training parameters using the following CLI arguments:

### Graph Generation (`generate_data.py`)
- `--n_products` (default: 200): Number of product nodes to generate.
- `--n_users` (default: 80): Number of user nodes to generate.
- `--n_sellers` (default: 20): Number of seller nodes to generate.
- `--anomaly_fraction` (default: 0.15): Ratio of anomalous nodes.
- `--seed` (default: 42): Random seed for reproducibility.
- `--output` (default: `data/graph.pt`): Where to save the generated graph.

### Training & Evaluation (`run_phase2.py` / `train_large.py`)
- `--graph_path` (default: `data/graph.pt`): Path to the generated graph file.
- `--gae_epochs` (default: 200): Total epochs for Stage 1 Unsupervised Training.
- `--gat_epochs` (default: 100): Total epochs for Stage 2 Semi-Supervised Training.
- `--embed_dim` (default: 128): Size of the node embeddings from the Stage 1 GAE.
- `--hidden_dim` (default: 64): Internal hidden dimension size for the Stage 2 GAT.
- `--lr` (default: 1e-3): Learning rate for both models.
- `--lam` (default: 0.5): Lambda hyperparameter controlling the balance between the supervised loss and the unsupervised smoothness penalty.
- `--results_dir` (default: `results`): Directory to save plots.
- `--skip_baselines`: Add this flag to `run_phase2.py` to skip executing the PyTorch Geometric reference baseline.
