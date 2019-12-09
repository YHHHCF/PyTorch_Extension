#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
  template <typename scalar_t>
  __global__ void horder_cuda_forward_kernel(
      torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> Z,  // (B, C, H, W)
      const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> x,  // (B, C, H, W)
      const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> weights,  // (k_size ^ 2, B, 1, H, W)
      const int k_size) {

    int h = blockIdx.x;
    int w = threadIdx.x;

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    for (int b = 0; b < B; b++) {
      for (int c = 0; c < C; c++) {
        // sweep the area around (h, w)
        for (int i = 0; i < k_size * k_size; i++) {
          int px = h + int(i / k_size) - 2;
          int py = w + int(i % k_size) - 2;
          if (px >= 0 && px < H && py >= 0 && py < W) {
            Z[b][c][h][w] += x[b][c][px][py] * weights[i][b][0][h][w];
          }
        }
      }
    }

  }

  template <typename scalar_t>
  __global__ void horder_cuda_backward_kernel(
      torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_x,  // (B, C, H, W)
      torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> d_weights,  // (kernel_size^2, B, 1, H, W)
      const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_Z,  // (B, C, H, W)
      const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> x,  // (B, C, H, W)
      const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> weights,  // (kernel_size^2, B, 1, H, W)
      const int k_size) {

    int h = blockIdx.x;
    int w = threadIdx.x;

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    for (int b = 0; b < B; b++) {
      for (int c = 0; c < C; c++) {
        // backprop through area at x affected by position i at weights
        for (int i = 0; i < k_size * k_size; i++) {
          int px = h + int(i / k_size) - 2;
          int py = w + int(i % k_size) - 2;
          int r_px = h + 2 - int(i / k_size);
          int r_py = w + 2 - int(i % k_size);

          if (px >= 0 && px < H && py >= 0 && py < W) {
            // the next formula is correct but has race
            // d_x[b][c][px][py] += grad_Z[b][c][h][w] * weights[i][b][0][h][w];
            d_weights[i][b][0][h][w] += grad_Z[b][c][h][w] * x[b][c][px][py];
          }

          if (r_px >= 0 && r_px < H && r_py >= 0 && r_py < W) {
            d_x[b][c][h][w] += grad_Z[b][c][r_px][r_py] * weights[i][b][0][r_px][r_py];
          }
        }
      }
    }

  }
} // namespace

torch::Tensor horder_cuda_forward(
    torch::Tensor x,
    torch::Tensor weights,
    int k_size) {

  auto Z = torch::zeros_like(x);  // (B, C, H, W)
  const int H = at::size(x, 2);
  const int W = at::size(x, 3);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "horder_forward_cuda", ([&] {
    horder_cuda_forward_kernel<scalar_t><<<H, W>>>(
        Z.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        weights.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        k_size);
  }));

  return Z;
}

std::vector<torch::Tensor> horder_cuda_backward(
    torch::Tensor grad_Z,
    torch::Tensor x,
    torch::Tensor weights,
    int k_size) {

  const int H = at::size(x, 2);
  const int W = at::size(x, 3);

  auto d_x = torch::zeros_like(x);  // (B, C, H, W)
  auto d_weights = torch::zeros_like(weights);  // (kernel_size^2, B, 1, H, W)

  // AT_DISPATCH_FLOATING_TYPES is an aten host function which launches kernel function
  // argument 1: x.type() specifies the tensor type in kernel function inputs
  // argument 2: a string to describe what this function does, for debug
  // argument 3: an anonymous function which calls the kernel function
  AT_DISPATCH_FLOATING_TYPES(x.type(), "horder_backward_cuda", ([&] {
    horder_cuda_backward_kernel<scalar_t><<<H, W>>>(
        // packed_accessor is a function that transfers a tensor from/to host to/from device
        // argument 1: scalar_t is they data element type of the tensor
        // argument 2: #dim of the tensor
        d_x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        d_weights.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        grad_Z.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        x.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        weights.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        k_size);
  }));

  return {d_x, d_weights};
}
