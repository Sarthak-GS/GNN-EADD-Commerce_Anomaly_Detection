/*
 * =============================================================================
 *  GNN-EADD Phase 2 — Custom CUDA Kernels for Parallel GNN Operations
 * =============================================================================
 *
 *  This file contains FOUR custom CUDA kernels targeting the computational
 *  bottlenecks identified in the GNN-EADD paper:
 *
 *    Kernel 1: smoothness_loss_kernel      — Unsupervised Loss (Eq. 17)
 *    Kernel 2: gat_attention_kernel        — Per-Edge Attention (Eq. 12)
 *    Kernel 3: neighbor_aggregation_kernel — Message Passing   (Eq. 7/13)
 *    Kernel 4: tiled_matmul_kernel         — Feature Transform (Tiled MM)
 *
 *  ──────────────────────────────────────────────────────────────────────
 *  LECTURE CONCEPTS DEMONSTRATED:
 *
 *  [Lec 1]  Motivation: Irregular graph structures (variable-degree nodes)
 *           create load imbalance and non-coalesced access patterns that
 *           demand careful parallel design.
 *
 *  [Lec 2]  Architecture: These kernels run on the GPU (SIMT model) and
 *           are compared against an OpenMP shared-memory CPU baseline.
 *
 *  [Lec 4]  GPGPU Computing: General-purpose computation on GPU hardware
 *           applied to Graph Neural Network inference/training.
 *
 *  [Lec 6-7]  Thread Organization: Each kernel uses a thread-per-edge (or
 *             thread-per-element) model with 256 threads/block.
 *             Grid size = ceil(work_items / 256).
 *
 *  [Lec 9,11] SIMT Execution: Kernels are designed to minimize warp
 *             divergence.  All threads in a warp follow the same path
 *             (boundary guard is the only branch).
 *
 *  [Lec 11,19] Memory Hierarchy: Global memory reads use __restrict__
 *              qualifiers for compiler hints.  Tiled matmul uses shared
 *              memory to amortize global loads.
 *
 *  [Lec 12]   Shared Memory: Kernel 4 (tiled matmul) loads tiles of A
 *             and B into __shared__ arrays, reducing global memory
 *             traffic by a factor of TILE_SIZE.
 *
 *  [Lec 14]   Parallel Reduction: Kernel 1 uses warp-level shuffle
 *             reduction (__shfl_down_sync) + shared memory reduction
 *             across warps + atomicAdd for final block-level accumulation.
 *
 *  [Lec 16-17] Tiled Matrix Multiplication: Kernel 4 decomposes the
 *              M×K × K×N multiplication into TILE_SIZE² sub-problems
 *              loaded into shared memory.
 *
 *  [Lec 20]   Synchronization: __syncthreads() for shared memory
 *             consistency; atomicAdd for many-to-one write patterns
 *             in neighbor aggregation and reduction kernels.
 *  ──────────────────────────────────────────────────────────────────────
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

/* ═══════════════════════════════════════════════════════════════════════════
 *  CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════ */

// [Lec 6-7] Thread block size: 256 threads is a standard choice that
// gives good occupancy on most NVIDIA GPUs (multiple of warp size 32).
// For 8.9M edges this yields ~34,800 blocks — well within grid limits.
constexpr int BLOCK_SIZE = 256;

// [Lec 16-17] Tile dimension for shared-memory matrix multiplication.
// 16×16 = 256 threads per block, matching BLOCK_SIZE for consistency.
// Each tile loads 16×16 floats = 1 KB into shared memory per matrix.
constexpr int TILE_SIZE = 16;


/* ═══════════════════════════════════════════════════════════════════════════
 *  KERNEL 1: UNSUPERVISED SMOOTHNESS LOSS  (Eq. 17)
 *  ─────────────────────────────────────────────────────────────────────────
 *   L_unsup = Σ_{(i,j) ∈ E}  A_ij * ||s_i - s_j||²
 *
 *   [Lec 6-7]  Thread mapping: one thread per edge in the sparse graph.
 *   [Lec 14]   Parallel Reduction: each thread computes one squared diff,
 *              then we reduce within warps using __shfl_down_sync,
 *              across warps using shared memory, and across blocks
 *              using atomicAdd on global memory.
 *   [Lec 20]   Synchronization: __syncthreads between warp-level and
 *              block-level reduction; atomicAdd for final accumulation.
 *   [Lec 9,11] SIMT: The only branch is the boundary guard (idx < E),
 *              so all threads in an active warp execute the same path.
 * ═══════════════════════════════════════════════════════════════════════════ */

template <typename scalar_t>
__global__ void smoothness_loss_kernel(
    const scalar_t* __restrict__ scores,    // [N] anomaly scores
    const int64_t*  __restrict__ row,       // [E] source node indices
    const int64_t*  __restrict__ col,       // [E] destination node indices
    scalar_t*       __restrict__ result,    // [1] output loss (atomicAdd)
    const int E)                            // number of edges
{
    /* [Lec 12] Shared memory for block-level reduction.
     * We need one slot per warp in the block (BLOCK_SIZE/32 = 8 warps). */
    __shared__ scalar_t warp_sums[BLOCK_SIZE / 32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ── Step 1: Per-thread computation (edge-parallel) ──
    // [Lec 6-7] Each thread handles exactly one edge.
    scalar_t val = 0.0;
    if (idx < E) {
        int64_t i = row[idx];
        int64_t j = col[idx];
        scalar_t diff = scores[i] - scores[j];
        val = diff * diff;  // ||s_i - s_j||²
    }

    // ── Step 2: Warp-level reduction using shuffle instructions ──
    // [Lec 14] Parallel Reduction — Warp Shuffle
    // __shfl_down_sync allows threads within the same warp (32 threads)
    // to directly exchange register values without shared memory,
    // halving the active threads at each step: 16 → 8 → 4 → 2 → 1.
    unsigned mask = 0xFFFFFFFF;  // all 32 lanes participate
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    // ── Step 3: Store warp results to shared memory ──
    // [Lec 20] Only lane 0 of each warp writes to shared memory.
    int lane   = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    if (lane == 0) {
        warp_sums[warpId] = val;
    }

    // [Lec 20] Barrier: ensure all warps have written before reading
    __syncthreads();

    // ── Step 4: First warp reduces the warp_sums array ──
    // [Lec 14] Second level of reduction across warps
    int numWarps = blockDim.x / 32;
    if (threadIdx.x < numWarps) {
        val = warp_sums[threadIdx.x];
    } else {
        val = 0.0;
    }
    if (warpId == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
    }

    // ── Step 5: Block result → global accumulator via atomicAdd ──
    // [Lec 20] Synchronization: atomicAdd prevents race conditions
    // when multiple blocks contribute to the same scalar result.
    if (threadIdx.x == 0) {
        atomicAdd(result, val);
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  KERNEL 2: PER-EDGE GAT ATTENTION SCORE  (Eq. 12)
 *  ─────────────────────────────────────────────────────────────────────────
 *   e_ij = LeakyReLU( a_r^T [W_r h_i || W_r h_j] )
 *
 *   [Lec 6-7]  One thread per edge: thread idx maps to edge (i,j).
 *   [Lec 11,19] Memory Coalescing: consecutive threads access
 *               consecutive edges, so row[idx] and col[idx] are
 *               coalesced reads from contiguous arrays.
 *               The Wh reads are scattered (indexed by node id) —
 *               this is inherent to irregular graph structure [Lec 1].
 *   [Lec 9,11] SIMT: the LeakyReLU ternary is branch-free via
 *              multiplication (val > 0 ? val : val * slope), which
 *              the compiler converts to a predicated instruction,
 *              avoiding warp divergence.
 * ═══════════════════════════════════════════════════════════════════════════ */

template <typename scalar_t>
__global__ void gat_attention_kernel(
    const scalar_t* __restrict__ Wh,          // [N, D] transformed features
    const int64_t*  __restrict__ row,         // [E] source node indices
    const int64_t*  __restrict__ col,         // [E] destination node indices
    const scalar_t* __restrict__ attn_vec,    // [2*D] attention vector a_r
    scalar_t*       __restrict__ e_out,       // [E] output attention logits
    const int E,                              // number of edges
    const int D,                              // feature dimension (out_dim)
    const float negative_slope)               // LeakyReLU parameter
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // [Lec 6-7] Boundary guard: standard pattern for mapping
    // a 1D grid of threads to a 1D array of work items (edges).
    if (idx >= E) return;

    int64_t src = row[idx];      // source node index
    int64_t dst = col[idx];      // destination node index

    // [Lec 11,19] Pointer arithmetic for row-major access.
    // Wh is stored as [N, D] so node i's features start at Wh + i*D.
    const scalar_t* Wh_src = Wh + src * D;
    const scalar_t* Wh_dst = Wh + dst * D;

    // Compute e_ij = a_r^T [Wh_dst || Wh_src]
    // First D elements of attn_vec multiply with Wh_dst (target node i),
    // Next D elements multiply with Wh_src (source node j).
    scalar_t val = 0.0;
    for (int d = 0; d < D; d++) {
        val += Wh_dst[d] * attn_vec[d];         // a_r[0:D] * Wh_i
        val += Wh_src[d] * attn_vec[D + d];     // a_r[D:2D] * Wh_j
    }

    // [Lec 9,11] LeakyReLU — branchless formulation to avoid
    // warp divergence across threads.
    e_out[idx] = (val > 0) ? val : val * static_cast<scalar_t>(negative_slope);
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  KERNEL 3: NEIGHBOR AGGREGATION / MESSAGE PASSING  (Eq. 7 / Eq. 13)
 *  ─────────────────────────────────────────────────────────────────────────
 *   out[dst] += alpha_ij * Wh[src]   for each edge (src→dst)
 *
 *   This is the "scatter" pattern: multiple edges may write to the
 *   same destination node (many-to-one), requiring synchronization.
 *
 *   [Lec 6-7]  Thread mapping: one thread per (edge, feature_dim) pair.
 *              Total threads = E × D, where E = edges, D = feature dim.
 *   [Lec 20]   Synchronization: atomicAdd on output[dst][d] handles the
 *              race condition when multiple source edges update the same
 *              destination node simultaneously.
 *   [Lec 1]    Challenge: high-degree nodes receive contributions from
 *              many edges → contention on atomicAdd.  This is the
 *              fundamental load-imbalance issue in graph parallelism.
 *   [Lec 9,11] SIMT: All threads execute the same instruction sequence;
 *              the only divergence point is the boundary guard.
 * ═══════════════════════════════════════════════════════════════════════════ */

template <typename scalar_t>
__global__ void neighbor_aggregation_kernel(
    const scalar_t* __restrict__ Wh,          // [N, D] source features
    const scalar_t* __restrict__ alpha,       // [E] normalized attention weights
    const int64_t*  __restrict__ row,         // [E] source node indices
    const int64_t*  __restrict__ col,         // [E] destination node indices
    scalar_t*       __restrict__ out,         // [N, D] output (zero-initialized)
    const int E,                              // number of edges
    const int D)                              // feature dimension
{
    // [Lec 6-7] Linearized 2D indexing: map thread to (edge, dim) pair.
    // This ensures consecutive threads in a warp process consecutive
    // feature dimensions of the same edge → better register utilization.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = E * D;

    if (tid >= total) return;

    int edge_id = tid / D;    // which edge
    int d       = tid % D;    // which feature dimension

    int64_t src = row[edge_id];
    int64_t dst = col[edge_id];

    // Contribution of this edge to destination node's feature d:
    //   out[dst][d] += alpha[edge] * Wh[src][d]
    scalar_t contribution = alpha[edge_id] * Wh[src * D + d];

    // [Lec 20] atomicAdd: multiple edges may target the same (dst, d),
    // so we need atomic operations to avoid write-after-write races.
    // This is the core synchronization challenge in graph message passing.
    atomicAdd(&out[dst * D + d], contribution);
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  KERNEL 4: TILED MATRIX MULTIPLICATION WITH SHARED MEMORY
 *  ─────────────────────────────────────────────────────────────────────────
 *   C[M,N] = A[M,K] × B[K,N]
 *   Used for feature transformation: H_out = H_in × W^T
 *
 *   [Lec 16-17] Tiled Matrix Multiplication: Instead of each thread
 *               reading an entire row of A and column of B from global
 *               memory (O(K) global reads per thread), we load TILE_SIZE
 *               elements at a time into shared memory, compute partial
 *               dot products, and iterate over tiles.
 *
 *   [Lec 12]    Shared Memory: Two __shared__ arrays (tile_A, tile_B)
 *               of size TILE_SIZE × TILE_SIZE each = 2 × 16 × 16 × 4
 *               = 2 KB per block.  This reduces global memory traffic
 *               by a factor of TILE_SIZE (16×).
 *
 *   [Lec 11,19] Memory Coalescing: Within each tile load, threads in
 *               the same row load consecutive columns → coalesced reads.
 *
 *   [Lec 20]    Synchronization: __syncthreads() after loading each
 *               tile ensures all threads have finished writing to shared
 *               memory before any thread begins reading from it.
 *
 *   [Lec 3]     Performance: Reduces global memory accesses from
 *               O(M×N×K) to O(M×N×K / TILE_SIZE), giving up to
 *               TILE_SIZE× speedup in memory-bound scenarios.
 * ═══════════════════════════════════════════════════════════════════════════ */

template <typename scalar_t>
__global__ void tiled_matmul_kernel(
    const scalar_t* __restrict__ A,   // [M, K]
    const scalar_t* __restrict__ B,   // [K, N]
    scalar_t*       __restrict__ C,   // [M, N]
    const int M, const int K, const int N)
{
    // [Lec 12] Allocate shared memory tiles for A and B sub-matrices
    __shared__ scalar_t tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_B[TILE_SIZE][TILE_SIZE];

    // [Lec 6-7] 2D thread indexing within the block
    int tx = threadIdx.x;   // column within tile
    int ty = threadIdx.y;   // row within tile

    // Global output position this thread is responsible for
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    scalar_t sum = 0.0;

    // Number of tiles needed to cover the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // ── Load tile from A[row, t*TILE + tx] ──
        // [Lec 11,19] Coalesced: threads with consecutive tx read
        // consecutive columns of A in the same row.
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            tile_A[ty][tx] = A[row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0;  // zero-pad out-of-bounds
        }

        // ── Load tile from B[t*TILE + ty, col] ──
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            tile_B[ty][tx] = B[b_row * N + col];
        } else {
            tile_B[ty][tx] = 0.0;
        }

        // [Lec 20] Barrier: all threads must finish loading the tile
        // before any thread starts computing with it.
        __syncthreads();

        // ── Compute partial dot product from this tile ──
        // [Lec 16-17] Each thread computes TILE_SIZE multiply-adds
        // using data from shared memory instead of global memory.
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        // [Lec 20] Barrier: ensure computation is done before next
        // iteration overwrites the shared memory tiles.
        __syncthreads();
    }

    // ── Write result to global memory ──
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  C++ WRAPPER FUNCTIONS (called from PyTorch C++ extension binding)
 *  ─────────────────────────────────────────────────────────────────────────
 *  Each wrapper:
 *    1. Validates tensor types and devices
 *    2. Computes grid/block dimensions  [Lec 6-7]
 *    3. Launches the CUDA kernel
 *    4. Returns PyTorch tensors
 * ═══════════════════════════════════════════════════════════════════════════ */

// ── Kernel 1 wrapper: Smoothness Loss ──────────────────────────────────────
torch::Tensor smoothness_loss_cuda(
    torch::Tensor scores,       // [N]
    torch::Tensor row,          // [E]
    torch::Tensor col)          // [E]
{
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA tensor");
    TORCH_CHECK(row.is_cuda(),    "row must be CUDA tensor");
    TORCH_CHECK(col.is_cuda(),    "col must be CUDA tensor");

    scores = scores.contiguous();
    row    = row.contiguous();
    col    = col.contiguous();

    int E = row.size(0);

    // [Lec 14] Output: single scalar initialized to 0 for reduction
    auto result = torch::zeros({1}, scores.options());

    // [Lec 6-7] Grid configuration: ceil(E / 256) blocks × 256 threads
    const int threads = BLOCK_SIZE;
    const int blocks  = (E + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(scores.scalar_type(), "smoothness_loss_cuda", ([&] {
        smoothness_loss_kernel<scalar_t><<<blocks, threads>>>(
            scores.data_ptr<scalar_t>(),
            row.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(),
            result.data_ptr<scalar_t>(),
            E
        );
    }));

    // Return mean (consistent with Phase 1 implementation)
    return result / static_cast<float>(E);
}


// ── Kernel 2 wrapper: GAT Attention Scores ─────────────────────────────────
torch::Tensor gat_attention_cuda(
    torch::Tensor Wh,           // [N, D]
    torch::Tensor row,          // [E]
    torch::Tensor col,          // [E]
    torch::Tensor attn_vec,     // [2*D]
    float negative_slope)
{
    TORCH_CHECK(Wh.is_cuda(),       "Wh must be CUDA tensor");
    TORCH_CHECK(row.is_cuda(),      "row must be CUDA tensor");
    TORCH_CHECK(col.is_cuda(),      "col must be CUDA tensor");
    TORCH_CHECK(attn_vec.is_cuda(), "attn_vec must be CUDA tensor");

    Wh       = Wh.contiguous();
    row      = row.contiguous();
    col      = col.contiguous();
    attn_vec = attn_vec.contiguous();

    int E = row.size(0);
    int D = Wh.size(1);

    auto e_out = torch::zeros({E}, Wh.options());

    // [Lec 6-7] One thread per edge
    const int threads = BLOCK_SIZE;
    const int blocks  = (E + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(Wh.scalar_type(), "gat_attention_cuda", ([&] {
        gat_attention_kernel<scalar_t><<<blocks, threads>>>(
            Wh.data_ptr<scalar_t>(),
            row.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(),
            attn_vec.data_ptr<scalar_t>(),
            e_out.data_ptr<scalar_t>(),
            E, D, negative_slope
        );
    }));

    return e_out;
}


// ── Kernel 3 wrapper: Neighbor Aggregation ─────────────────────────────────
torch::Tensor neighbor_aggregation_cuda(
    torch::Tensor Wh,           // [N, D]
    torch::Tensor alpha,        // [E]
    torch::Tensor row,          // [E]
    torch::Tensor col,          // [E]
    int num_nodes)              // N
{
    TORCH_CHECK(Wh.is_cuda(),    "Wh must be CUDA tensor");
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA tensor");
    TORCH_CHECK(row.is_cuda(),   "row must be CUDA tensor");
    TORCH_CHECK(col.is_cuda(),   "col must be CUDA tensor");

    Wh    = Wh.contiguous();
    alpha = alpha.contiguous();
    row   = row.contiguous();
    col   = col.contiguous();

    int E = row.size(0);
    int D = Wh.size(1);

    // Output initialized to zeros; atomicAdd accumulates contributions
    auto out = torch::zeros({num_nodes, D}, Wh.options());

    // [Lec 6-7] Total threads = E × D (one per edge-feature pair)
    int total = E * D;
    const int threads = BLOCK_SIZE;
    const int blocks  = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(Wh.scalar_type(), "neighbor_aggregation_cuda", ([&] {
        neighbor_aggregation_kernel<scalar_t><<<blocks, threads>>>(
            Wh.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            row.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            E, D
        );
    }));

    return out;
}


// ── Kernel 4 wrapper: Tiled Matrix Multiplication ──────────────────────────
torch::Tensor tiled_matmul_cuda(
    torch::Tensor A,    // [M, K]
    torch::Tensor B)    // [K, N]
{
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // [Lec 6-7] 2D grid of TILE_SIZE×TILE_SIZE thread blocks
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "tiled_matmul_cuda", ([&] {
        tiled_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    }));

    return C;
}
