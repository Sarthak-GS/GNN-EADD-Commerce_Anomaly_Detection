#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void gat_attention_kernel(
    const scalar_t* __restrict__ Wh,
    const int64_t* __restrict__ edge_index,
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ e,
    int E, 
    int out_dim, 
    float negative_slope) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        int64_t src = edge_index[0 * E + idx];
        int64_t dst = edge_index[1 * E + idx];
        
        scalar_t val = 0.0;
        // Compute dot product of a_r^T [Wh_i || Wh_j]
        // Wh_dst is target node (i), Wh_src is source node (j)
        for (int i = 0; i < out_dim; i++) {
            val += Wh[dst * out_dim + i] * a[i];
            val += Wh[src * out_dim + i] * a[out_dim + i];
        }
        
        // LeakyReLU Activation
        e[idx] = val > 0 ? val : val * static_cast<scalar_t>(negative_slope);
    }
}

torch::Tensor gat_attention_cuda_forward(
    torch::Tensor Wh,
    torch::Tensor edge_index,
    torch::Tensor a,
    float negative_slope) 
{
    int E = edge_index.size(1);
    int out_dim = Wh.size(1);
    
    // Output tensor for attention logits
    auto e = torch::zeros({E}, Wh.options());
    
    const int threads = 256;
    const int blocks = (E + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(Wh.scalar_type(), "gat_attention_cuda_forward", ([&] {
        gat_attention_kernel<scalar_t><<<blocks, threads>>>(
            Wh.data_ptr<scalar_t>(),
            edge_index.data_ptr<int64_t>(),
            a.data_ptr<scalar_t>(),
            e.data_ptr<scalar_t>(),
            E, 
            out_dim, 
            negative_slope
        );
    }));
    
    return e;
}
