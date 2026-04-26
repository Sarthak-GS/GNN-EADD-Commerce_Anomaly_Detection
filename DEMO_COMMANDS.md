# GNN-EADD Phase 2: Live Demo & Execution Guide

## 👨‍🏫 LIVE PRESENTATION GUIDE (LARGE GRAPH)
Because training on a large graph takes time, you should train the model *before* the presentation. During the live demo, you will only show the **Inference Speed Comparison** and the **Performance Benchmarks**.

### PHASE A: Prep Work (Do this before the presentation)
```bash
# 1. Activate Environment
conda activate sarthak_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
cd ~/Desktop/POP/project/gnn_eadd

# 2. Build Kernels (if not done already)
pip install -e . --no-build-isolation

# 3. Generate a LARGE Graph (e.g., 5000 products, 2000 users, 500 sellers)
# This creates data/graph.pt with 7,500 total nodes and many edges.
python generate_data.py --n_products 5000 --n_users 2000 --n_sellers 500

# 4. Train the Model (This takes time, let it finish)
# We skip benchmarks here, just getting the weights saved to results/checkpoint.pt
python run_phase2.py --skip_benchmark --skip_baselines
```

### PHASE B: The Live Demo (Do this in front of the Professor)
When the professor asks to see it work, do NOT run training again. Run these exact commands to show off the speed:

```bash
# 1. Show the Raw Kernel Benchmarks (The 2900x CPU Speedup)
# This tests the C++ kernels against pure math on 1k, 10k, and 50k nodes.
python benchmark.py --results_dir results

# 2. Show the PyTorch vs CUDA Inference Comparison (The 46x GPU Speedup)
# This loads your saved large graph and runs inference TWICE (PyTorch vs CUDA)
# It will prove identical accuracy but massive speedups.
# Add --skip_training so it only does the inference phase using the saved weights!
python run_phase2.py --skip_training --skip_benchmark --skip_baselines
```

---

## Alternative / Utility Commands

```bash
# Run absolutely everything (Slowest)
python run_phase2.py

# Just OpenMP CPU smoke test:
python openmp_baseline.py

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
| `train_large.py` | Phase 1 training pipeline with PyG Mini-Batching (GAE Stage 1 + GAT Stage 2) |
| `baseline_comparison.py` | PyTorch Geometric (PyG) dual-stage baseline for accuracy comparison |
| `visualize.py` | Generates loss curves, anomaly score distributions, ROC/PR curves, graph visualization |

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
