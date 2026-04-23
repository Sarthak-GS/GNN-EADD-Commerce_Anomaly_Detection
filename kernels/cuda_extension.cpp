#include <torch/extension.h>

/**
 * GNN-EADD C++ Bindings for CUDA Kernels
 */

// Forward declarations of CUDA wrappers
torch::Tensor smoothness_loss_cuda(torch::Tensor s, torch::Tensor edge_index);
torch::Tensor attention_scores_cuda(torch::Tensor Wh, torch::Tensor edge_index, torch::Tensor a, float leaky_slope);
torch::Tensor neighbor_agg_cuda(torch::Tensor msgs, torch::Tensor dst_idx, int num_nodes);

// API Interface functions (with device checks)
torch::Tensor smoothness_loss(torch::Tensor s, torch::Tensor edge_index) {
    return smoothness_loss_cuda(s, edge_index);
}

torch::Tensor attention_scores(torch::Tensor Wh, torch::Tensor edge_index, torch::Tensor a, float leaky_slope) {
    return attention_scores_cuda(Wh, edge_index, a, leaky_slope);
}

torch::Tensor neighbor_agg(torch::Tensor msgs, torch::Tensor dst_idx, int num_nodes) {
    return neighbor_agg_cuda(msgs, dst_idx, num_nodes);
}

// Module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smoothness_loss", &smoothness_loss, "CUDA Smoothness Loss");
    m.def("attention_scores", &attention_scores, "CUDA Attention Scores");
    m.def("neighbor_agg", &neighbor_agg, "CUDA Neighbor Aggregation");
}
