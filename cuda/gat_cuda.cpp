#include <torch/extension.h>
#include <stdexcept>

// Forward declaration of the CUDA kernel driver function
torch::Tensor gat_attention_cuda_forward(
    torch::Tensor Wh,
    torch::Tensor edge_index,
    torch::Tensor a,
    float negative_slope);

// The C++ interface that PyTorch calls
torch::Tensor gat_cuda_forward(
    torch::Tensor Wh,
    torch::Tensor edge_index,
    torch::Tensor a,
    float negative_slope) 
{
    // Type and device checks
    TORCH_CHECK(Wh.device().is_cuda(), "Wh must be a CUDA tensor");
    TORCH_CHECK(edge_index.device().is_cuda(), "edge_index must be a CUDA tensor");
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    
    // Ensure contiguous memory mapping for C pointers
    Wh = Wh.contiguous();
    edge_index = edge_index.contiguous();
    a = a.contiguous();

    // Route purely to the CUDA kernel dispatcher
    return gat_attention_cuda_forward(Wh, edge_index, a, negative_slope);
}

// Bind the C++ function into the Python namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gat_cuda_forward, "GAT custom CUDA forward pass");
}
