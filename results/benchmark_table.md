# GNN-EADD Phase 2 — Benchmark Results

> [Lec 3] Amdahl's Law: S(N) = 1 / ((1-P) + P/N)
> where P = parallel fraction, N = number of processors


## Smoothness Loss

### Execution Times

| Size | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Small (1K) | 1.812 ms | 0.023 ms | 0.026 ms |
| Medium (10K) | 17.833 ms | 0.036 ms | 0.023 ms |
| Large (50K) | N/A | 0.095 ms | 0.019 ms |

### Speedup vs Sequential

| Size | OpenMP Speedup | CUDA Speedup |
|---|---|---|
| Small (1K) | 77.8× | 68.8× |
| Medium (10K) | 499.4× | 780.2× |
| Large (50K) | N/A | N/A |

## GAT Attention

### Execution Times

| Size | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Small (1K) | 164.567 ms | 0.192 ms | 0.034 ms |
| Medium (10K) | 1258.180 ms | 0.502 ms | 0.069 ms |
| Large (50K) | N/A | 2.163 ms | 0.492 ms |

### Speedup vs Sequential

| Size | OpenMP Speedup | CUDA Speedup |
|---|---|---|
| Small (1K) | 857.7× | 4850.3× |
| Medium (10K) | 2508.8× | 18144.2× |
| Large (50K) | N/A | N/A |

## Neighbor Aggregation

### Execution Times

| Size | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Small (1K) | 134.573 ms | 0.486 ms | 0.019 ms |
| Medium (10K) | 1041.982 ms | 1.957 ms | 0.035 ms |
| Large (50K) | N/A | 9.655 ms | 0.245 ms |

### Speedup vs Sequential

| Size | OpenMP Speedup | CUDA Speedup |
|---|---|---|
| Small (1K) | 276.8× | 7209.7× |
| Medium (10K) | 532.5× | 29618.0× |
| Large (50K) | N/A | N/A |

## Matrix Multiply

### Execution Times

| Size | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Small (1K) | N/A | 0.087 ms | 0.017 ms |
| Medium (10K) | N/A | 0.064 ms | 0.012 ms |
| Large (50K) | N/A | 0.050 ms | 0.012 ms |


## Amdahl's Law Analysis [Lec 3]

For each operation, we estimate the parallel fraction P from measured speedup S
using: P = (1 - 1/S) / (1 - 1/N)

| Operation | Measured CUDA Speedup | Est. Parallel Fraction P | Theoretical Max Speedup (∞ cores) |
|---|---|---|---|
| Smoothness Loss | 780.2× | 1.0000 | inf× |
| GAT Attention | 18144.2× | 1.0000 | inf× |
| Neighbor Aggregation | 29618.0× | 1.0000 | inf× |
| Matrix Multiply | N/A | N/A | N/A |


## Hardware Configuration

| Resource | Specification |
|---|---|
| GPU | NVIDIA RTX A6000 |
| GPU Memory | 47.4 GB |
| Compute Capability | 8.6 |
| CPU Cores | 24 |
| CUDA Version | 12.1 |
| PyTorch Version | 2.2.1+cu121 |