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
Install dependencies:
```bash
pip install -r requirements.txt
```

Run the complete pipeline:
```bash
python run_all.py
```
