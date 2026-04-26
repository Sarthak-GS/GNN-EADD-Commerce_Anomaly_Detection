# GNN-EADD: E-Commerce Anomaly Detection via Dual-Stage Learning

This repository implements GNN-EADD, a Graph Neural Network framework designed for anomaly detection in heterogeneous e-commerce environments. It utilizes a two-stage approach: unsupervised representation learning via Graph Auto-Encoders (GAE) and semi-supervised classification via Graph Attention Networks (GAT).

## Project Architecture

The codebase is organized into several key modules:

- **root/**: Core training and execution entry points.
- **models/**: Implementation of GAE and GAT architectures using type-specific layers.
- **data/**: Scripts for synthetic graph generation and real-world data processing.
- **scripts/**: Evaluation tools for benchmarking, baselines, and plotting.
- **utils/**: Shared mathematical utilities and performance metrics.
- **kernels/**: Custom CUDA and OpenMP parallel operations for acceleration.

## Getting Started

### 1. Installation
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

Compile the custom CUDA kernels for hardware acceleration:
```bash
pip install -e .
```

### 2. Data Preparation
Generate a synthetic heterogeneous graph with products, users, and sellers:
```bash
python data/generate_data.py --n_products 500 --n_users 200 --output data/graph.pt
```

For real-world datasets (e.g., Amazon All Beauty), use the processing script:
```bash
python data/process_real_data.py
```

### 3. Training the Model
Train the dual-stage pipeline on the generated graph:
```bash
python train.py --graph_path data/graph.pt
```

For large-scale graphs (e.g., >100k nodes), use the memory-efficient mini-batch trainer:
```bash
python train_large.py --graph_path data/real_graph.pt --batch_size 1024
```

### 4. Performance Benchmarking
Compare the execution times of Sequential, OpenMP, and CUDA implementations:
```bash
python scripts/benchmark.py
```

Run the master pipeline to execute all tests, benchmarks, and training in one go:
```bash
python run_phase2.py --graph_path data/graph.pt
```

### 5. Visualization and Analysis
Generate loss curves, ROC/PR curves, and graph structure plots:
```bash
python scripts/visualize.py
```

Compare results against professional 3rd-party GNN frameworks (PyG):
```bash
python scripts/baseline_comparison.py
```

## Key Features

- **Dual-Stage Learning**: Decouples representation learning from anomaly classification for stability.
- **Heterogeneous Support**: Handles multiple node and edge types with specific attention mechanisms.
- **Hardware Acceleration**: Custom CUDA kernels optimized for sparse graph operations.
- **Scalability**: Support for mini-batch training enables processing of massive e-commerce datasets.

## Note on Custom Data
To use your own data, ensure it follows the HeteroGraph structure defined in `data/graph_builder.py` and save it as a `.pt` file compatible with the training scripts.
