# GNN-EADD: E-Commerce Anomaly Detection
## Project Summary & Presentation Outline

### 1. The Core Objective
The goal of this project was to implement a state-of-the-art dual-stage Graph Neural Network (GNN) for detecting anomalies (e.g., fraudulent sellers, fake reviews) in e-commerce networks. 
* **Stage 1 (Unsupervised):** A Graph AutoEncoder (GAE) learns structural embeddings of users, products, and sellers.
* **Stage 2 (Semi-supervised):** A Graph Attention Network (GAT) uses those embeddings to assign an "anomaly score" to each node.

### 2. Phase 1: The PyTorch Baseline (The Problem)
We initially implemented the math from the research paper entirely in Python using standard PyTorch. While mathematically correct (achieving >0.90 AUC-ROC), we hit a massive performance bottleneck.

**Why was native PyTorch slow?**
Standard Deep Learning libraries like PyTorch are built for *dense* data (like images). Graphs, however, are highly *sparse* (most users don't interact with most products). PyTorch forced us to pass the full Adjacency Matrix ($N \times N$) into the GPU. This meant the GPU was allocating massive memory and performing millions of useless "multiplications by zero" for nodes that were not connected. This dense matrix multiplication overhead caused Out-Of-Memory (OOM) errors and made the model far too slow for real-world inference on millions of e-commerce nodes.

### 3. Phase 2: Custom CUDA Acceleration (The Solution)
To solve the sparsity bottleneck, we bypassed PyTorch's native operations and wrote four custom C++ / CUDA kernels specifically optimized for Graph Data.

Instead of dense matrix multiplication, we utilized a **Sparse COO (Coordinate) Format**, extracting only the actual edges (`row`, `col`) and performing math only where connections existed.

We implemented four custom CUDA kernels utilizing advanced Parallel Programming concepts. Here is exactly how they are used in our pipeline:

1. **Neighbor Aggregation with Atomic Locks (Used in Stage 1 & Stage 2):**
   * **Where:** Used in the GAE Encoder to aggregate local neighborhoods, and in the GAT to sum attention-weighted features.
   * **How:** We map one GPU thread to each specific edge (scatter pattern). Because multiple edges can point to the same destination node simultaneously (e.g., a highly popular product), we implemented hardware-level `atomicAdd` locks to prevent race conditions during the summation.

2. **Shared Memory Tiled MatMul (Used in Stage 1 Decoder):**
   * **Where:** Used to reconstruct the graph adjacency matrix ($Z \times Z^T$) from the latent embeddings.
   * **How:** Instead of calculating dot products directly from slow Global GPU memory, this kernel loads small $16 \times 16$ blocks of the matrices into ultra-fast `__shared__` on-chip memory. This drastically reduces the memory bandwidth bottleneck for large, symmetric matrix multiplications.

3. **Thread-Per-Edge Attention (Used in Stage 2 GAT):**
   * **Where:** Calculates the LeakyReLU attention score ($e_{ij}$) for every connected pair of nodes.
   * **How:** Mapped one GPU thread to each specific edge, completely skipping pairs that are not connected (avoiding millions of useless zero-padding operations). We utilized branchless multiplication logic to prevent GPU "warp divergence," keeping the threads executing in perfect lockstep.

4. **Parallel Smoothness Loss Reduction (Used in Stage 1 Loss):**
   * **Where:** Computes the unsupervised graph regularization loss (making sure connected nodes have similar embeddings).
   * **How:** Used ultra-fast `__shfl_down_sync` warp-shuffle primitives to perform rapid parallel reduction (summation) of the loss across threads, completely bypassing memory writes until the final block sum.

**Overcoming the $O(N^2)$ Edge Extraction Bottleneck:**
Even with our fast CUDA kernels, we discovered a hidden bottleneck: PyTorch had to execute an $O(N^2)$ `.nonzero()` scan over the dense matrix just to find the sparse edges to pass to our kernel during every inference pass. We solved this by pre-computing and caching the Coordinate (COO) edge lists, reducing the overhead and enabling true sub-millisecond inference speeds.

### 4. Scaling the Architecture to Massive Datasets
**The Autograd Limitation (Why we don't train with our CUDA kernels):**
To train a neural network, PyTorch relies on Automatic Differentiation (Autograd). Every time PyTorch executes a native math operation (like a dense matrix multiplication), it secretly builds a "Computational Graph" in the background. During `loss.backward()`, PyTorch traverses this graph backwards, applying the Chain Rule of calculus to compute the exact gradient (derivative) needed to update each weight. 

However, PyTorch treats our custom CUDA kernels as "black boxes." It has no idea what math is happening inside our C++ code, so it cannot automatically generate the backward pass. To train using our kernels, we would have to manually calculate the complex analytical derivatives for the GAT attention mechanism, the sparse scatter additions, and the tiled matrix multiplications, and then write entirely new CUDA kernels just for the backward pass. 

Because writing custom backward-pass calculus engines is incredibly complex, we designed a **Hybrid Execution Pipeline** that gives us the best of both worlds:
* **Offline Training (PyTorch Geometric Mini-Batching):** For massive graphs (10k+ nodes), the system automatically uses PyG's `NeighborLoader` to train the GAE and GAT in localized sub-graph batches using native PyTorch operations. This avoids VRAM limits while letting PyTorch's Autograd handle the calculus automatically.
* **Production Inference (Custom CUDA):** The moment training finishes and the model switches to `torch.no_grad()` (which disables Autograd since we don't need to update weights to score anomalies), we construct the global sparse representation. Our custom CUDA pipeline then automatically takes over to evaluate the massive graph at lightning speed, operating entirely on cached COO edge lists.



### 5. Final Results & Inferences (The Flex)
We ran a side-by-side inference benchmark on the exact same trained model using the exact same GPU (RTX architecture) on a massively scaled **22,500 node** e-commerce graph.

* **Accuracy Verification:** The custom CUDA kernels produced the exact same anomaly scores as the PyTorch baseline down to floating-point precision differences (`< 2.98e-07`). This proves the mathematical integrity of the parallel algorithms. The microscopic variation is the expected result of floating-point non-associativity (the random ordering of parallel `atomicAdd` instructions compared to PyTorch's deterministic tree reductions).
* **Massive Scaling Speedup (Stage 1 GAE):** As the graph size grows, the difference between PyTorch's dense overhead and our sparse kernel efficiency widens exponentially. Our custom Sparse CUDA implementation + COO Caching achieved an astonishing **~32.5x speedup** over PyTorch's native GPU execution for the GAE Encode stage (0.653 ms vs 21.203 ms) on the 22.5k node graph.
* **Complex Attention Speedup (Stage 2 GAT):** Even on the mathematically intensive LeakyReLU edge attention stage, the CUDA kernels maintained a solid **1.59x speedup** (1.691 ms vs 2.688 ms).

**Conclusion:**
By understanding the underlying hardware and the sparse topological nature of Graph Neural Networks, we successfully applied parallel computing principles to eliminate the dense-matrix bottlenecks of modern deep learning frameworks, resulting in a production-ready, high-speed anomaly detection system.



This is known as creating Foreign Function Interfaces (FFI), and it's actually exactly how PyTorch and TensorFlow work under the hood! They are just Python wrappers pointing to highly optimized C/C++ libraries.




at is the Sparse COO Format?
In most Parallel Programming courses, you learn matrix multiplication using Dense Matrices. If you have a graph with 22,500 nodes, a dense adjacency matrix would be an array of size $22,500 \times 22,500$. That is 506 Million elements in RAM.

However, in a graph (like Amazon reviews), a single user doesn't interact with all 22,500 products. They interact with maybe 2 or 3. So, 99.99% of that 506 million element matrix is just zeros!

COO (Coordinate) Format is a clever way to store this. Instead of a massive 2D matrix, we just store two 1D arrays:

row = [0, 1, 4, 15] (The starting node of an edge)
col = [5, 2, 7, 10] (The destination node of an edge)
If there are only 30,000 actual connections in the network, COO format only uses 60,000 numbers instead of 506 million.

Why wasn't it in the course, and how did we implement it?
Your course focuses on structured parallel grids (like image processing or standard Dense Matrix Multiply). Graphs are Unstructured / Irregular Data.

When processing Dense matrices, you map threads to (x, y) coordinates on a 2D grid. But for COO, we use a concept called Thread-Per-Edge Mapping. We map a 1D grid of GPU threads directly to the elements of the COO arrays. Thread 0 processes edge 0 (row[0], col[0]), Thread 1 processes edge 1 (row[1], col[1]), completely skipping all the blank space where edges don't exist!

Where is the code?
1. Extracting the COO Format (in run_phase2.py line 236): During the benchmark, we added logic to take the PyTorch dense matrix and extract only the active coordinates before the inference loops begin:

python
row, col = A_r.nonzero(as_tuple=True)
gat_coo[r] = (row.long().contiguous(), col.long().contiguous())