#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
  __global__ void horder_cuda_forward_kernel(
      float* Z,  // (B, C, H, W)
      float* x,  // (B, C, H, W)
      float* weights,  // (k_size ^ 2, B, 1, H, W)
      int B,
      int C,
      int H,
      int W,
      int k_size) {
    // const int n = blockIdx.y;
    // const int c = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.x;
    int w = threadIdx.x;

    // printf("thread: %d, %d\n", i, j);

    // if (h == 0 && w == 0) {
    //   printf("1st number x: %f\n", x[0]);
    //   printf("2nd number x: %f\n", x[1]);

    //   printf("1st number w: %f\n", weights[0]);
    //   printf("2nd number w: %f\n", weights[1]);
    // }

    for (int b = 0; b < B; b++) {
      for (int c = 0; c < C; c++) {
        int idx_z = b * C * H * W + c * H * W + h * W + w;

        // sweep the area around (h, w)
        for (int i = 0; i < k_size; i++) {
          for (int j = 0; j < k_size; j++) {
            int px = h + i - 2;
            int py = w + j - 2;
            if (px >= 0 && px < H && py >= 0 && py < W) {
              int idx_x = b * C * H * W + c * H * W + px * W + py;
              int idx_weights = (i * k_size + j) * B * H * W + b * H * W + h * W + w;

              // Z[b][c][h][w] += x[b][c][px][py] * weights[i, j][b][0][h][w]
              Z[idx_z] += x[idx_x] + weights[idx_weights];
            }
          }
        }
      }
    }

    // if (h==0 && w==0) {
    //   printf("Z gpu: %f\n", Z[0]);
    // }
  }

  __global__ void horder_cuda_backward_kernel(
      float* d_x,  // (B, C, H + 4, W + 4)
      float* d_weights,  // (kernel_size^2, B, 1, H, W)
      float* grad_Z,  // (B, C, H, W)
      float* x,  // (B, C, H + 4, W + 4)
      float* weights,  // (kernel_size^2, B, 1, H, W)
      int B,
      int C,
      int H,
      int W,
      int k_size) {
    // const int n = blockIdx.y;
    // const int c = blockIdx.x * blockDim.x + threadIdx.x;

    printf("backward debug\n");

    int i = blockIdx.x;
    int j = threadIdx.x;

    int idx_weights = 0;

    // for (int i = 0; i < k_size; i++) {
    //   for (int j = 0; j < k_size; j++) {
        int idx_z = 0;
        int idx_x = i * (W + 4) + j;

        for (int b = 0; b < B; b++) {
          for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
              for (int w = 0; w < W; w++) {
                // d_x[b][c][h+i][w+j] += grad_Z[b][c][h][w] * weights[i * k_size + j][b][0][h][w]
                d_x[idx_x] += grad_Z[idx_z] * weights[idx_weights];

                // d_weights[i * k_size + j][b][0][h][w] += grad_Z[b][c][h][w] * x[b][c][h+i][w+j]
                d_weights[idx_weights] += grad_Z[idx_z] * x[idx_x];

                idx_z++;
                idx_x++;
                idx_weights++;
              }
              idx_z += W;
              idx_x += (W + 4);
              idx_weights += W;
            }
            idx_z += H * W;
            idx_x += (H + 4) * (W + 4);
            idx_weights += H * W;
          }
          idx_z += C * H * W;
          idx_x += C * (H + 4) * (W + 4);
          idx_weights += H * W;
        }
        idx_weights += B * H * W;
    //   }
    // }
  }
} // namespace

torch::Tensor horder_cuda_forward(
    torch::Tensor x,
    torch::Tensor weights,
    int k_size) {

  int B = at::size(x, 0);
  int C = at::size(x, 1);
  int H = at::size(x, 2);
  int W = at::size(x, 3);

  auto Z = torch::zeros_like(x);  // (B, C, H, W)
  // x = at::constant_pad_nd(x, torch::IntList{2, 2, 2, 2}, 0);  // (B, C, H + 4, W + 4)

  // size_t size_Z = B * C * H * W * sizeof(float);
  // size_t size_x = B * C * H * W * sizeof(float);
  // size_t size_weights = k_size * k_size * B * H * W * sizeof(float);

  // float* device_Z;
  // float* device_x;
  // float* device_weights;

  // std::cout<<"debug 1"<<std::endl;

  // cudaMalloc(&device_Z, size_Z);
  // cudaMalloc(&device_x, size_x);
  // cudaMalloc(&device_weights, size_weights);

  // cudaMemcpy(device_Z, Z.data<float>(), size_Z, cudaMemcpyHostToDevice);
  // cudaMemcpy(device_x, x.data<float>(), size_x, cudaMemcpyHostToDevice);
  // cudaMemcpy(device_weights, weights.data<float>(), size_weights, cudaMemcpyHostToDevice);

  // std::cout<<"debug 2"<<std::endl;

  horder_cuda_forward_kernel<<<H, W>>>(Z.data<float>(),
                                      x.data<float>(), weights.data<float>(),
                                       B, C, H, W, k_size);
  // auto tmp = Z;
  // std::cout << Z << std::endl;

  // std::cout<<"debug 3"<<std::endl;

  // // cudaMemcpy(Z.data<float>(), device_Z, size_Z, cudaMemcpyDeviceToHost);

  // std::cout<<"debug 4"<<std::endl;

  // cudaFree(device_Z);
  // cudaFree(device_x);
  // cudaFree(device_weights);

  // std::cout<<"debug 5"<<std::endl;

  return Z;
}

std::vector<torch::Tensor> horder_cuda_backward(
    torch::Tensor grad_Z,
    torch::Tensor x,
    torch::Tensor weights,
    int k_size) {

  int B = at::size(x, 0);
  int C = at::size(x, 1);
  int H = at::size(x, 2);
  int W = at::size(x, 3);

  x = at::constant_pad_nd(x, torch::IntList{2, 2, 2, 2}, 0);  // (B, C, H + 4, W + 4)
  auto d_x = torch::zeros_like(x);  // (B, C, H + 4, W + 4)
  auto d_weights = torch::zeros_like(weights);  // (kernel_size^2, B, 1, H, W)
  
  // be careful here, x is already padded
  horder_cuda_backward_kernel<<<k_size, k_size>>>(d_x.data<float>(), d_weights.data<float>(),
                                        grad_Z.data<float>(), x.data<float>(), weights.data<float>(),
                                        B, C, H, W, k_size);

  d_x = torch::slice(d_x, 2, 2, H + 2);
  d_x = torch::slice(d_x, 3, 2, W + 2);  // (B, C, H, W)

  return {d_x, d_weights};
}
