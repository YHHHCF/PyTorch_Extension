#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
  template <typename scalar_t>
  __global__ void horder_cuda_forward_kernel(
      torch::Tensor x,
      torch::Tensor weights,
      int H,
      int W,
      torch::Tensor Z) {

    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
  }

  template <typename scalar_t>
  __global__ void horder_cuda_backward_kernel(
      torch::Tensor grad_Z,
      torch::Tensor x,
      torch::Tensor weights,
      int k_size,
      torch::Tensor d_x,
      torch::Tensor d_weights,) {

    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    
  }
} // namespace

torch::Tensor horder_cuda_forward(
    torch::Tensor x,
    torch::Tensor weights,
    int H,
    int W) {

  auto Z = torch::zeros_like(x);
  
  horder_cuda_forward_kernel<<<1, 1>>>(x, weights, H, W, Z);
  return Z;
}

std::vector<torch::Tensor> horder_cuda_backward(
    torch::Tensor grad_Z,
    torch::Tensor x,
    torch::Tensor weights,
    int k_size) {

  int H = at::size(x, 2);
  int W = at::size(x, 3);

  x = at::constant_pad_nd(x, torch::IntList{2, 2, 2, 2}, 0);  // (B, C, H + 4, W + 4)

  auto d_x = torch::zeros_like(x);  // (B, C, H + 4, W + 4)
  auto d_weights = torch::zeros_like(weights);  // (kernel_size^2, B, 1, H, W)
  
  // be careful here, x is already padded
  horder_cuda_backward_kernel<<<1, 1>>>(grad_Z, x, weights, k_size, d_x, d_weights);
  return {d_x, d_weights};
}
