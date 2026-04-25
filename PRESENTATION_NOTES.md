# GNN-EADD: E-Commerce Anomaly Detection
## Project Summary & Presentation Outline

### 1. The Core Objective
The goal of this project was to implement a state-of-the-art dual-stage Graph Neural Network (GNN) for detecting anomalies (e.g., fraudulent sellers, fake reviews) in e-commerce networks. 
* **Stage 1 (Unsupervised):** A Graph AutoEncoder (GAE) learns structural embeddings of users, products, and sellers.
* **Stage 2 (Semi-supervised):** A Graph Attention Network (GAT) uses those embeddings to assign an "anomaly score" to each node.

### 2. Phase 1: The PyTorch Baseline (The Problem)
We initially implemented the math from the research paper entirely in Python using standard PyTorch. While mathematically correct (achieving >0.90 AUC-ROC), we hit a massive performance bottleneck.

**Why was native PyTorch slow?**
Standard Deep Learning libraries like PyTorch are built for *dense* data (like images). Graphs, however, are highly *sparse* (most users don't interact with most products). PyTorch forced us to pass the full Adjacency Matrix ($N \times N$) into the GPU. This meant the GPU was performing millions of useless "multiplications by zero" for nodes that were not connected. This dense matrix multiplication overhead made the model far too slow for real-world inference on millions of e-commerce nodes.

### 3. Phase 2: Custom CUDA Acceleration (The Solution)
To solve the sparsity bottleneck, we bypassed PyTorch's native operations and wrote four custom C++ / CUDA kernels specifically optimized for Graph Data.

Instead of dense matrix multiplication, we utilized a **Sparse COO (Coordinate) Format**, extracting only the actual edges (`row`, `col`) and performing math only where connections existed.

We implemented four key kernels utilizing advanced Parallel Programming concepts:
1. **Thread-Per-Edge Attention (GAT):** Mapped one GPU thread to each specific edge to compute LeakyReLU attention scores, completely avoiding zero-padding. Used branchless logic to prevent warp divergence.
2. **Neighbor Aggregation with Atomic Locks:** Used a thread-per-edge scatter pattern to aggregate messages. Implemented `atomicAdd` to resolve race conditions when multiple edges tried to update the same high-degree destination node simultaneously.
3. **Shared Memory Tiled MatMul:** For feature projection ($H \times W^T$), we utilized `__shared__` memory tiling to load blocks of the matrix into fast on-chip memory, drastically reducing global memory bandwidth bottleneck.
4. **Parallel Smoothness Loss Reduction:** Used `__shfl_down_sync` warp primitives to rapidly compute the unsupervised graph regularization loss.

### 4. The Engineering Workflow
Because writing custom Autograd engines (the backward pass calculus) is incredibly complex, we designed a hybrid execution pipeline:
* **Offline Training:** The system uses the native PyTorch dense operations, allowing Autograd to calculate gradients and update weights automatically. 
* **Production Inference:** The moment training finishes and the model switches to `torch.no_grad()` (evaluating the graph to find the actual anomalies), our custom CUDA pipeline automatically takes over to score the massive graph at lightning speed.

### 5. Final Results & Inferences (The Flex)
We ran a side-by-side inference benchmark on the exact same trained model using the exact same GPU (RTX architecture).

* **Accuracy Verification:** The custom CUDA kernels produced the exact same anomaly scores as the PyTorch baseline down to floating-point precision differences (`< 4.17e-07`). This proves the mathematical integrity of the parallel algorithms. The microscopic variation is the expected result of floating-point non-associativity (the random ordering of parallel `atomicAdd` instructions compared to PyTorch's deterministic tree reductions).
* **Speedup (GPU vs GPU):** Our custom Sparse CUDA implementation achieved a **46.4x speedup** over PyTorch's native dense GPU execution for the GAT Stage (1.78 ms vs 82.67 ms).
* **Speedup (CPU vs GPU):** Against the sequential CPU baseline on a 10,000-node benchmark, our kernels achieved a massive **2900x speedup** (0.39 ms vs 1177 ms).

**Conclusion:**
By understanding the underlying hardware and the sparse topological nature of Graph Neural Networks, we successfully applied parallel computing principles to eliminate the dense-matrix bottlenecks of modern deep learning frameworks, resulting in a production-ready, high-speed anomaly detection system.
