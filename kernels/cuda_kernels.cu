#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * GNN-EADD Raw CUDA Kernels
 * 
 * "Giving it the final touch" :)
 */

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 1: Smoothness Loss
// ─────────────────────────────────────────────────────────────────────────────

__global__ void smoothness_loss_kernel(
    const float* __restrict__ s,          // [N] node scores
    const long* __restrict__ src_idx,    // [E]
    const long* __restrict__ dst_idx,    // [E]
    float* __restrict__ total_out,       // [1] output sum
    int E
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < E) {
        float diff = s[src_idx[tid]] - s[dst_idx[tid]];
        atomicAdd(total_out, diff * diff);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 2: Attention Scores
// ─────────────────────────────────────────────────────────────────────────────

__global__ void attention_scores_kernel(
    const float* __restrict__ Wh,         // [N x D]
    const long* __restrict__ src_idx,    // [E]
    const long* __restrict__ dst_idx,    // [E]
    const float* __restrict__ a,          // [2 x D] attention vector
    float* __restrict__ out,              // [E]
    int E,
    int D,
    float leaky_slope
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < E) {
        long src = src_idx[tid];
        long dst = dst_idx[tid];
        
        float sum = 0.0f;
        // dot(a, [Wh_src || Wh_dst])
        for (int d = 0; d < D; d++) {
            sum += Wh[src * D + d] * a[d];
            sum += Wh[dst * D + d] * a[D + d];
        }
        
        // LeakyReLU
        out[tid] = (sum > 0.0f) ? sum : sum * leaky_slope;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL 3: Neighbor Aggregation
// ─────────────────────────────────────────────────────────────────────────────

__global__ void neighbor_agg_kernel(
    const float* __restrict__ msgs,       // [E x D]
    const long* __restrict__ dst_idx,    // [E]
    float* __restrict__ out,              // [N x D]
    int E,
    int D
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int edge_idx = tid / D;
    int dim_idx  = tid % D;

    if (edge_idx < E) {
        long dst = dst_idx[edge_idx];
        atomicAdd(&out[dst * D + dim_idx], msgs[tid]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WRAPPERS (To be called from C++)
// ─────────────────────────────────────────────────────────────────────────────

torch::Tensor smoothness_loss_cuda(
    torch::Tensor s,
    torch::Tensor edge_index
) {
    auto E = edge_index.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(s.device());
    auto total_out = torch::zeros({1}, options);

    const int threads = 256;
    const int blocks = (E + threads - 1) / threads;

    smoothness_loss_kernel<<<blocks, threads>>>(
        s.data_ptr<float>(),
        edge_index[0].data_ptr<long>(),
        edge_index[1].data_ptr<long>(),
        total_out.data_ptr<float>(),
        E
    );

    return total_out / (float)E; // Mean squared diff
}

torch::Tensor attention_scores_cuda(
    torch::Tensor Wh,
    torch::Tensor edge_index,
    torch::Tensor a,
    float leaky_slope
) {
    auto E = edge_index.size(1);
    auto D = Wh.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(Wh.device());
    auto out = torch::empty({E}, options);

    const int threads = 256;
    const int blocks = (E + threads - 1) / threads;

    attention_scores_kernel<<<blocks, threads>>>(
        Wh.data_ptr<float>(),
        edge_index[0].data_ptr<long>(),
        edge_index[1].data_ptr<long>(),
        a.data_ptr<float>(),
        out.data_ptr<float>(),
        E,
        D,
        leaky_slope
    );

    return out;
}

torch::Tensor neighbor_agg_cuda(
    torch::Tensor msgs,
    torch::Tensor dst_idx,
    int num_nodes
) {
    auto E = msgs.size(0);
    auto D = msgs.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(msgs.device());
    auto out = torch::zeros({num_nodes, (long)D}, options);

    const int threads = 256;
    const int blocks = (E * D + threads - 1) / threads;

    neighbor_agg_kernel<<<blocks, threads>>>(
        msgs.data_ptr<float>(),
        dst_idx.data_ptr<long>(),
        out.data_ptr<float>(),
        E,
        D
    );

    return out;
}
