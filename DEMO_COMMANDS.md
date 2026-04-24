# GNN-EADD Phase 2: Demo Commands & Execution Guide

## Quick Reference — Copy-Paste Commands

### Step 0: Activate Environment
```bash
conda activate sarthak_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
cd ~/Desktop/POP/project/gnn_eadd
```

### Step 1: Build Custom CUDA Kernels (one-time)
```bash
pip install -e . --no-build-isolation
```

### Step 2: Generate Synthetic Graph (small, for RTX 2050)
```bash
python generate_data.py --n_products 500 --n_users 200 --n_sellers 50
```

### Step 3: Run Everything (Training + Benchmarks + Baselines + Plots)
```bash
python run_phase2.py
```

### Step 3 (Alternative): Run Individual Steps
```bash
# Just run benchmarks (no training):
python run_phase2.py --skip_training --skip_baselines

# Just train + evaluate:
python run_phase2.py --skip_benchmark --skip_baselines

# Just benchmark:
python benchmark.py --results_dir results

# Just OpenMP smoke test:
python openmp_baseline.py

# Just kernel correctness tests:
python -c "from run_phase2 import test_kernel_correctness; test_kernel_correctness()"

# Just PyG baseline comparison:
python baseline_comparison.py

# Regenerate plots from existing checkpoint:
python visualize.py
```

---

## What Each Script Does

| Script | Purpose |
|--------|---------|
| `setup.py` | Compiles `csrc/kernels.cu` + `csrc/binding.cpp` into `gnn_cuda_kernels` Python module |
| `openmp_baseline.py` | Compiles + loads C/OpenMP shared library, provides CPU-parallel baselines |
| `benchmark.py` | Runs Sequential vs OpenMP vs CUDA timing, generates all performance plots |
| `run_phase2.py` | Master script: builds kernels → tests correctness → benchmarks → trains → baselines → plots |
| `train.py` | Phase 1 training pipeline (GAE Stage 1 + GAT Stage 2) |
| `baseline_comparison.py` | PyTorch Geometric (PyG) dual-stage baseline for accuracy comparison |
| `visualize.py` | Generates loss curves, anomaly score distributions, ROC/PR curves, graph visualization |
| `verify.py` | Checkpoint integrity verification (CPU-only) |

---

## Output Files (in `results/`)

### Performance Plots
| File | Shows |
|------|-------|
| `benchmark_times.png` | Log-scale execution times: Sequential vs OpenMP vs CUDA |
| `speedup_chart.png` | Speedup curves (Amdahl's Law) |
| `time_distribution_pie.png` | Computation time breakdown per operation |
| `resource_utilization.png` | GPU vs CPU hardware utilization |
| `accuracy_comparison.png` | PyG vs Custom CUDA accuracy (proves no degradation) |
| `benchmark_table.md` | Full numerical results + Amdahl's Law analysis |

### Model/Accuracy Plots
| File | Shows |
|------|-------|
| `loss_curves.png` | GAE + GAT training convergence |
| `anomaly_scores.png` | Score distribution: normal vs anomalous |
| `roc_pr_curves.png` | ROC and Precision-Recall curves |
| `graph_structure.png` | Graph visualization with anomaly highlighting |

---

## Lecture Concept Mapping

| Concept | Lecture | Where Demonstrated |
|---------|--------|--------------------|
| Irregular graph parallelization | Lec 1 | `kernels.cu` header comments, neighbor_aggregation kernel |
| Shared-memory vs accelerator | Lec 2 | `openmp_baseline.py` (CPU) vs `kernels.cu` (GPU) |
| Amdahl's Law & speedup | Lec 3 | `benchmark.py` → `benchmark_table.md`, `speedup_chart.png` |
| GPGPU for GNNs | Lec 4 | All kernels in `kernels.cu`, `setup.py` |
| Thread-per-edge mapping | Lec 6-7 | `BLOCK_SIZE=256`, grid config in all kernel wrappers |
| SIMT & warp divergence | Lec 9,11 | Branchless LeakyReLU in `gat_attention_kernel` |
| Memory coalescing | Lec 11,19 | Contiguous edge array reads, `__restrict__` qualifiers |
| Shared memory | Lec 12 | `tiled_matmul_kernel` tile_A/tile_B arrays |
| Parallel reduction | Lec 14 | `smoothness_loss_kernel` warp shuffle + shared mem reduction |
| Tiled matrix multiply | Lec 16-17 | `tiled_matmul_kernel` with TILE_SIZE=16 |
| Synchronization | Lec 20 | `atomicAdd` in aggregation/reduction, `__syncthreads` in matmul |

---

## For Amazon Electronics Dataset (Future — Better GPU)

When you get the real dataset and a more powerful GPU:

### Step 1: Prepare the dataset
Convert Amazon Electronics reviews into a `HeteroData` object with:
- Node types: `product` (498K), `user` (4.9M), `seller`
- Edge types: `('product', 'purchase', 'user')`, `('seller', 'sell', 'product')`, `('user', 'interact', 'user')`
- Save to `data/amazon_graph.pt`

### Step 2: Update setup.py for your GPU
```python
# For RTX 4090 (sm_89):
'-gencode=arch=compute_89,code=sm_89',

# For A100 (sm_80):
'-gencode=arch=compute_80,code=sm_80',
```
Then rebuild: `pip install -e . --no-build-isolation`

### Step 3: Run with large graph
```bash
python run_phase2.py --graph_path data/amazon_graph.pt --include_xlarge
```

The `--include_xlarge` flag adds a 200K node / 1M edge benchmark tier that
really shows the CUDA speedup advantage at scale.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'gnn_cuda_kernels'` | Run `pip install -e . --no-build-isolation` |
| `libc10.so: cannot open shared object file` | Run `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH` |
| `CUDA out of memory` | Reduce graph size in `generate_data.py` |
| `nvcc fatal: Unsupported gpu architecture 'compute_XX'` | Change `-gencode` in `setup.py` to match your GPU |
