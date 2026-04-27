/* CUDA kernels and wrappers for GNN phase-2 operations. */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

/* Launch configuration constants. */
constexpr int BLOCK_SIZE = 256;

constexpr int TILE_SIZE = 16;


/* Kernel 1: reduce edge-wise squared differences into one loss scalar. */

template <typename scalar_t>
__global__ void smoothness_loss_kernel(
    const scalar_t* __restrict__ scores,
    const int64_t*  __restrict__ row,
    const int64_t*  __restrict__ col,
    scalar_t*       __restrict__ result,
    const int E)
{
    /* One partial sum per warp for block-level reduction. */
    __shared__ scalar_t warp_sums[BLOCK_SIZE / 32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    scalar_t val = 0.0;
    if (idx < E) {
        int64_t i = row[idx];
        int64_t j = col[idx];
        scalar_t diff = scores[i] - scores[j];
        val = diff * diff;
    }

    unsigned mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    int lane   = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    if (lane == 0) {
        warp_sums[warpId] = val;
    }

    /* Ensure all warp sums are written before block reduction. */
    __syncthreads();

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

    /* Accumulate block result safely across blocks. */
    if (threadIdx.x == 0) {
        atomicAdd(result, val);
    }
}


/* Kernel 2: compute attention logits for each edge. */

template <typename scalar_t>
__global__ void gat_attention_kernel(
    const scalar_t* __restrict__ Wh,
    const int64_t*  __restrict__ row,
    const int64_t*  __restrict__ col,
    const scalar_t* __restrict__ attn_vec,
    scalar_t*       __restrict__ e_out,
    const int E,
    const int D,
    const float negative_slope)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= E) return;

    int64_t src = row[idx];
    int64_t dst = col[idx];

    const scalar_t* Wh_src = Wh + src * D;
    const scalar_t* Wh_dst = Wh + dst * D;

    scalar_t val = 0.0;
    for (int d = 0; d < D; d++) {
        val += Wh_dst[d] * attn_vec[d];
        val += Wh_src[d] * attn_vec[D + d];
    }

    e_out[idx] = (val > 0) ? val : val * static_cast<scalar_t>(negative_slope);
}


/* Kernel 3: scatter-add messages from source to destination nodes. */

template <typename scalar_t>
__global__ void neighbor_aggregation_kernel(
    const scalar_t* __restrict__ Wh,
    const scalar_t* __restrict__ alpha,
    const int64_t*  __restrict__ row,
    const int64_t*  __restrict__ col,
    scalar_t*       __restrict__ out,
    const int E,
    const int D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = E * D;

    if (tid >= total) return;

    int edge_id = tid / D;
    int d       = tid % D;

    int64_t src = row[edge_id];
    int64_t dst = col[edge_id];

    scalar_t contribution = alpha[edge_id] * Wh[src * D + d];

    /* Atomic add is required because many edges may target the same output. */
    atomicAdd(&out[dst * D + d], contribution);
}


/* Kernel 4: tiled matrix multiplication using shared memory. */

template <typename scalar_t>
__global__ void tiled_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t*       __restrict__ C,
    const int M, const int K, const int N)
{
    /* Shared-memory tiles for one A submatrix and one B submatrix. */
    __shared__ scalar_t tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t tile_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    scalar_t sum = 0.0;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            tile_A[ty][tx] = A[row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0;
        }

        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            tile_B[ty][tx] = B[b_row * N + col];
        } else {
            tile_B[ty][tx] = 0.0;
        }

        /* Synchronize after tile load before compute. */
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        /* Synchronize before loading the next tile. */
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


/* C++ wrapper functions exposed through the PyTorch extension binding. */

/* Wrapper for Kernel 1. */
torch::Tensor smoothness_loss_cuda(
    torch::Tensor scores,
    torch::Tensor row,
    torch::Tensor col)
{
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA tensor");
    TORCH_CHECK(row.is_cuda(),    "row must be CUDA tensor");
    TORCH_CHECK(col.is_cuda(),    "col must be CUDA tensor");

    scores = scores.contiguous();
    row    = row.contiguous();
    col    = col.contiguous();

    int E = row.size(0);

    auto result = torch::zeros({1}, scores.options());

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

    return result / static_cast<float>(E);
}


/* Wrapper for Kernel 2. */
torch::Tensor gat_attention_cuda(
    torch::Tensor Wh,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor attn_vec,
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


/* Wrapper for Kernel 3. */
torch::Tensor neighbor_aggregation_cuda(
    torch::Tensor Wh,
    torch::Tensor alpha,
    torch::Tensor row,
    torch::Tensor col,
    int num_nodes)
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

    /* Output starts at zero and is accumulated with atomicAdd in the kernel. */
    auto out = torch::zeros({num_nodes, D}, Wh.options());

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


/* Wrapper for Kernel 4. */
torch::Tensor tiled_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B)
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
