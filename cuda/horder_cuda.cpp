#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> horder_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    int H,
    int W);

std::vector<torch::Tensor> horder_cuda_backward(
    torch::Tensor grad_Z,
    torch::Tensor x,
    torch::Tensor weights,
    int k_size);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> horder_forward(
    torch::Tensor x,
    torch::Tensor weights,
    int H, int W) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  return horder_cuda_forward(input, weights, H, W);
}

std::vector<torch::Tensor> horder_backward(
    torch::Tensor grad_Z,  // (B, C, H, W)
    torch::Tensor x,
    torch::Tensor weights,
    int k_size) {
  CHECK_INPUT(grad_Z);
  CHECK_INPUT(x);
  CHECK_INPUT(weights);

  return horder_cuda_backward(grad_Z, x, weights, k_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &horder_forward, "High Order forward (CUDA)");
  m.def("backward", &horder_backward, "High Order backward (CUDA)");
}
