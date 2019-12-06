#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> horder_forward(
    torch::Tensor x,
    torch::Tensor weights,
    int H, int W) {

  auto Z = torch::zeros_like(x);
  int k = 0;
  x = torch::constant_pad_2d(x, torch::IntList{2, 2, 2, 2}, 0);

  for (int hh = 0; hh < 5; hh++) {
    for (int ww = 0; ww < 5; ww++) {
      auto X_c = torch::slice(x, 2, hh, H + hh);
      auto X_c = torch::slice(X_c, 3, ww, W + ww);
      Z += X_c * weights[k];
      k++;
    }
  }

  return Z;
}

std::vector<torch::Tensor> horder_backward(
    torch::Tensor grad_Z,  // (B, C, H, W)
    torch::Tensor x,
    torch::Tensor weights,
    int B, int C, int H, int W, int k_size) {
  
  auto d_x = torch::zeros_like(x);  // (B, C, H, W)
  auto d_weights = torch::zeros_like(weights);  // (kernel_size^2, B, 1, H, W)

  x = torch::constant_pad_2d(x, torch::IntList{2, 2, 2, 2}, 0);

  // calculate d_x and d_weights
  for (int b = 0; b < B; b++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          for (int i = 0; i < k_size; i++) {
            for (int j = 0; j < k_size; j++) {
              int px = h + i - 2;
              int py = w + j - 2;
              if (px >= 0 && px < H && py >= 0 && py < W) {
                d_x[b][c][px][py] += grad_Z[b][c][h][w] * weights[i * k_size + j][b][0][h][w];
                d_weights[i * k_size + j][b][0][h][w] += grad_Z[b][c][h][w] * d_x[b][c][px][py];
              }
            }
          }
        }
      }
    }
  }

  return {d_x, d_weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &horder_forward, "High Order forward");
  m.def("backward", &horder_backward, "High Order backward");
}
