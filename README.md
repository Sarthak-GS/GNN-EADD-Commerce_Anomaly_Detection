# GNN-EADD: E-Commerce Anomaly Detection

Detects anomalous behaviour in e-commerce platforms using a dual-stage heterogeneous Graph Neural Network. Phase 2 adds GPU-parallel CUDA kernels, an OpenMP CPU baseline, an asymmetric MLP decoder, and a full timing benchmark.

## Overview

1. Stage 1 (Unsupervised): Graph Auto-Encoder (GAE) learns node embeddings from graph structure. Supports symmetric inner-product decoder or asymmetric MLP decoder.
2. Stage 2 (Semi-Supervised): Type-specific GAT classifies anomalies using frozen embeddings + a small label set. Smoothness loss computed via thread-per-edge parallel kernel (Kernel 1).

## File Structure

```
models/
  gae.py              Stage 1 GAE (inner_product or mlp decoder)
  gat.py              Stage 2 GAT
  mlp_decoder.py      Asymmetric MLP decoder for directed edges

kernels/
  cuda_ops.py         3 thread-per-edge GPU-compatible kernels
  openmp_ops.py       OpenMP CPU wrappers (with PyTorch fallback)
  openmp_kernels.c    C source compiled to openmp_ext.so
  build_ext.py        Compile script for OpenMP extension

data/graph_builder.py Synthetic heterogeneous graph construction
train.py              Dual-stage training loop (all modes)
run_all.py            Full pipeline orchestrator
benchmark.py          Kernel timing comparison table
baseline_comparison.py PyTorch Geometric reference baseline
visualize.py          Loss curves, anomaly distributions, ROC/PR plots
```

## Setup

```bash
conda activate gnn_anomaly
pip install -r requirements.txt

# Optional: compile the OpenMP C extension (requires gcc with -fopenmp)
python kernels/build_ext.py
```

## Quick Start

```bash
# Generate synthetic graph
python generate_data.py --output data/graph.pt

# Sequential (Phase 1 baseline)
python train.py

# CUDA-mode parallel kernel (Kernel 1 for smoothness loss)
python train.py --parallel_mode cuda --decoder_type mlp

# OpenMP CPU parallel (4 threads)
python train.py --parallel_mode openmp --n_threads 4

# Full pipeline with visualization + baselines
python run_all.py --parallel_mode cuda --decoder_type mlp

# Kernel timing benchmark
python benchmark.py --n_threads 4
```

## Parallel Modes

| Flag | Kernel | Description |
|---|---|---|
| `--parallel_mode sequential` | Dense N x N ops | Default baseline |
| `--parallel_mode cuda` | Sparse thread-per-edge | GPU when available, CPU fallback |
| `--parallel_mode openmp` | Sparse thread-per-edge | C extension with OpenMP, n_threads configurable |

The three target kernels from the paper:
- Kernel 1: Smoothness loss `L_unsup = sum_{(i,j) in E} ||s_i - s_j||^2` — used in GAT training
- Kernel 2: Per-edge attention scores `e_ij = LeakyReLU(a^T [Wh_i || Wh_j])` — benchmarked
- Kernel 3: Neighbor aggregation `h_v = sum_{u in N(v)} coeff * W * h_u` — benchmarked

## Decoder Types

| Flag | Description |
|---|---|
| `--decoder_type inner_product` | Symmetric `Z Z^T` (default, Phase 1) |
| `--decoder_type mlp` | Asymmetric 2-layer MLP on directed edge pairs |

## Key Arguments (`train.py` / `run_all.py`)

- `--decoder_type {inner_product,mlp}` — GAE decoder (default: inner_product)
- `--parallel_mode {sequential,openmp,cuda}` — Kernel execution mode (default: sequential)
- `--n_threads N` — OpenMP thread count (default: 4)
- `--gae_epochs N` — Stage 1 epochs (default: 200)
- `--gat_epochs N` — Stage 2 epochs (default: 100)
- `--embed_dim N` — Node embedding size (default: 128)
- `--hidden_dim N` — Hidden layer size (default: 64)
- `--graph_path PATH` — Path to graph file (default: data/graph.pt)
- `--results_dir PATH` — Output directory (default: results/)
