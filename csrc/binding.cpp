/*
 * =============================================================================
 *  binding.cpp — PyTorch C++ Extension Binding for GNN-EADD CUDA Kernels
 * =============================================================================
 *
 *  [Lec 2] Programming Model: This file bridges the CUDA kernels (GPU code)
 *  with the Python training loop via PyTorch's C++ extension mechanism.
 *  The CUDA kernels operate on raw device pointers while PyTorch handles
 *  memory management, autograd, and device placement transparently.
 *
 *  After compilation (python setup.py install), these functions become
 *  callable in Python as:
 *      import gnn_cuda_kernels
 *      loss = gnn_cuda_kernels.smoothness_loss(scores, row, col)
 *      attn = gnn_cuda_kernels.gat_attention(Wh, row, col, a, slope)
 *      out  = gnn_cuda_kernels.neighbor_aggregation(Wh, alpha, row, col, N)
 *      C    = gnn_cuda_kernels.tiled_matmul(A, B)
 */

#include <torch/extension.h>

// Forward declarations of CUDA wrapper functions defined in kernels.cu
torch::Tensor smoothness_loss_cuda(
    torch::Tensor scores,
    torch::Tensor row,
    torch::Tensor col);

torch::Tensor gat_attention_cuda(
    torch::Tensor Wh,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor attn_vec,
    float negative_slope);

torch::Tensor neighbor_aggregation_cuda(
    torch::Tensor Wh,
    torch::Tensor alpha,
    torch::Tensor row,
    torch::Tensor col,
    int num_nodes);

torch::Tensor tiled_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B);


// ── Python-facing wrappers with input validation ──────────────────────────

torch::Tensor smoothness_loss(
    torch::Tensor scores,
    torch::Tensor row,
    torch::Tensor col)
{
    TORCH_CHECK(scores.device().is_cuda(), "scores must be a CUDA tensor");
    TORCH_CHECK(row.device().is_cuda(),    "row must be a CUDA tensor");
    TORCH_CHECK(col.device().is_cuda(),    "col must be a CUDA tensor");
    return smoothness_loss_cuda(scores.contiguous(), row.contiguous(), col.contiguous());
}

torch::Tensor gat_attention(
    torch::Tensor Wh,
    torch::Tensor row,
    torch::Tensor col,
    torch::Tensor attn_vec,
    float negative_slope)
{
    TORCH_CHECK(Wh.device().is_cuda(),       "Wh must be a CUDA tensor");
    TORCH_CHECK(row.device().is_cuda(),      "row must be a CUDA tensor");
    TORCH_CHECK(col.device().is_cuda(),      "col must be a CUDA tensor");
    TORCH_CHECK(attn_vec.device().is_cuda(), "attn_vec must be a CUDA tensor");
    return gat_attention_cuda(
        Wh.contiguous(), row.contiguous(), col.contiguous(),
        attn_vec.contiguous(), negative_slope);
}

torch::Tensor neighbor_aggregation(
    torch::Tensor Wh,
    torch::Tensor alpha,
    torch::Tensor row,
    torch::Tensor col,
    int num_nodes)
{
    TORCH_CHECK(Wh.device().is_cuda(),    "Wh must be a CUDA tensor");
    TORCH_CHECK(alpha.device().is_cuda(), "alpha must be a CUDA tensor");
    TORCH_CHECK(row.device().is_cuda(),   "row must be a CUDA tensor");
    TORCH_CHECK(col.device().is_cuda(),   "col must be a CUDA tensor");
    return neighbor_aggregation_cuda(
        Wh.contiguous(), alpha.contiguous(), row.contiguous(), col.contiguous(),
        num_nodes);
}

torch::Tensor tiled_matmul(
    torch::Tensor A,
    torch::Tensor B)
{
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    return tiled_matmul_cuda(A.contiguous(), B.contiguous());
}


// ── Register functions into the Python module ─────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("smoothness_loss",        &smoothness_loss,
          "Unsupervised smoothness loss (CUDA, Eq.17)");
    m.def("gat_attention",          &gat_attention,
          "Per-edge GAT attention scores (CUDA, Eq.12)");
    m.def("neighbor_aggregation",   &neighbor_aggregation,
          "Neighbor aggregation / message passing (CUDA, Eq.7/13)");
    m.def("tiled_matmul",           &tiled_matmul,
          "Tiled matrix multiplication with shared memory (CUDA)");
}
