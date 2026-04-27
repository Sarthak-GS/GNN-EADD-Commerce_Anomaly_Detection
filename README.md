# GNN-EADD

GNN-EADD is a graph neural network pipeline for anomaly detection in heterogeneous e-commerce graphs. It uses two stages:

1. Graph Auto-Encoder (GAE) for representation learning.
2. Graph Attention Network (GAT) for anomaly classification.

## Repository Layout

- [data/](data/) - synthetic data generation and real-data preprocessing.
- [models/](models/) - GAE and GAT model definitions.
- [scripts/](scripts/) - benchmarking, baselines, and visualisation utilities.
- [utils/](utils/) - shared helpers and metrics.
- [csrc/](csrc/) and [build_openmp/](build_openmp/) - custom CUDA and OpenMP kernels.

## Requirements

Before installing, make sure your PyTorch build matches the CUDA version on your machine.

- If your system CUDA toolkit is `cu118`, install a PyTorch wheel built for CUDA 11.8.
- If your system CUDA toolkit is `cu121`, install a PyTorch wheel built for CUDA 12.1.
- Do not mix a PyTorch CPU wheel with CUDA extension builds.
- Rebuild the extension after changing either PyTorch or CUDA versions.

Recommended checks:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

## Installation

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

Then build the CUDA extension in editable mode:

```bash
pip install -e . --no-build-isolation
```

If you use a GPU, verify that `torch.version.cuda` matches the CUDA toolchain used to build the extension.

## Data Preparation

Generate a synthetic graph:

```bash
python data/generate_data.py --n_products 500 --n_users 200 --output data/graph.pt
```

## Training

 4. Train the Model (This takes time, let it finish)
 We skip benchmarks here, just getting the weights saved to results/checkpoint.pt
 ```bash
python run_phase2.py --skip_benchmark --graph_path data/graph.pt
```


Train the large-graph variant:

```bash
python train_large.py --graph_path data/real_graph.pt --batch_size 1024
```

Run the full phase-2 flow:

```bash
python run_phase2.py --graph_path data/graph.pt
```


if already trained

```bash
python run_phase2.py --skip_training --graph_path data/graph.pt
```

## Benchmarking

Compare sequential, OpenMP, and CUDA execution:

```bash
python scripts/benchmark.py
```

Run baseline comparisons:

```bash
python scripts/baseline_comparison.py
```

## Visualisation

Generate the plots saved under `results/`:

```bash
python scripts/visualize.py
```

## Custom Data

Custom datasets should follow the heterogeneous graph structure defined in [data/graph_builder.py](data/graph_builder.py) and be saved as a `.pt` file compatible with the training scripts.
