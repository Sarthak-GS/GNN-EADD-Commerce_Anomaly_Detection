# GNN-EADD Phase 2 — Benchmark Results

> [Lec 3] Amdahl's Law: S(N) = 1 / ((1-P) + P/N)
> where P = parallel fraction, N = number of processors


## Smoothness Loss

### Execution Times

| Size | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Small (1K) | 1.502 ms | 0.013 ms | 0.025 ms |
| Medium (10K) | 13.036 ms | 0.022 ms | 0.019 ms |
| Large (50K) | N/A | 0.099 ms | 0.076 ms |

### Speedup vs Sequential

| Size | OpenMP Speedup | CUDA Speedup |
|---|---|---|
| Small (1K) | 116.7× | 61.2× |
| Medium (10K) | 596.9× | 669.3× |
| Large (50K) | N/A | N/A |

## GAT Attention

### Execution Times

| Size | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Small (1K) | 106.681 ms | 0.081 ms | 0.074 ms |
| Medium (10K) | 1076.549 ms | 0.673 ms | 1.056 ms |
| Large (50K) | N/A | 3.811 ms | 13.725 ms |

### Speedup vs Sequential

| Size | OpenMP Speedup | CUDA Speedup |
|---|---|---|
| Small (1K) | 1320.5× | 1442.3× |
| Medium (10K) | 1600.0× | 1019.0× |
| Large (50K) | N/A | N/A |

## Neighbor Aggregation

### Execution Times

| Size | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Small (1K) | 115.387 ms | 0.295 ms | 0.025 ms |
| Medium (10K) | 1177.147 ms | 2.608 ms | 0.393 ms |
| Large (50K) | N/A | 13.531 ms | 2.289 ms |

### Speedup vs Sequential

| Size | OpenMP Speedup | CUDA Speedup |
|---|---|---|
| Small (1K) | 391.2× | 4669.6× |
| Medium (10K) | 451.3× | 2995.1× |
| Large (50K) | N/A | N/A |

## Matrix Multiply

### Execution Times

| Size | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Small (1K) | N/A | 0.061 ms | 0.024 ms |
| Medium (10K) | N/A | 0.064 ms | 0.025 ms |
| Large (50K) | N/A | 0.071 ms | 0.029 ms |


## Amdahl's Law Analysis [Lec 3]

For each operation, we estimate the parallel fraction P from measured speedup S
using: P = (1 - 1/S) / (1 - 1/N)

| Operation | Measured CUDA Speedup | Est. Parallel Fraction P | Theoretical Max Speedup (∞ cores) |
|---|---|---|---|
| Smoothness Loss | 669.3× | 1.0000 | inf× |
| GAT Attention | 1019.0× | 1.0000 | inf× |
| Neighbor Aggregation | 2995.1× | 1.0000 | inf× |
| Matrix Multiply | N/A | N/A | N/A |


## Hardware Configuration

| Resource | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 2050 |
| GPU Memory | 3.7 GB |
| Compute Capability | 8.6 |
| CPU Cores | 16 |
| CUDA Version | 12.1 |
| PyTorch Version | 2.5.1+cu121 |