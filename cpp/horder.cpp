#include <torch/extension.h>
#include <vector>

torch::Tensor horder_forward(
    torch::Tensor x,
    torch::Tensor weights,
    int H, int W) {

  auto Z = torch::zeros_like(x);
  int k = 0;
  x = at::constant_pad_nd(x, torch::IntList{2, 2, 2, 2}, 0);

  for (int hh = 0; hh < 5; hh++) {
    for (int ww = 0; ww < 5; ww++) {
      auto X_c = torch::slice(x, 2, hh, H + hh);
      X_c = torch::slice(X_c, 3, ww, W + ww);
      Z += at::mul(X_c, weights[k]);
      k++;
    }
  }
  return Z;
}

std::vector<torch::Tensor> horder_backward(
    torch::Tensor grad_Z,  // (B, C, H, W)
    torch::Tensor x,
    torch::Tensor weights,
    int k_size) {

  int H = at::size(x, 2);
  int W = at::size(x, 3);

  x = at::constant_pad_nd(x, torch::IntList{2, 2, 2, 2}, 0);  // (B, C, H + 4, W + 4)

  auto d_x = torch::zeros_like(x);  // (B, C, H + 4, W + 4)
  auto d_weights = torch::zeros_like(weights);  // (kernel_size^2, B, 1, H, W)

  int pad_size = k_size - 1;

  for (int i = 0; i < k_size; i++) {
    for (int j = 0; j < k_size; j++) {
      auto grad_x = at::mul(grad_Z, weights[i * k_size + j]);
      d_x += at::constant_pad_nd(grad_x, torch::IntList{j, pad_size - j, i, pad_size - i}, 0);  // left, right, up, down
      auto slice_x = torch::slice(x, 2, i, H + i);
      slice_x = torch::slice(slice_x, 3, j, W + j);

      auto res = at::sum(at::mul(grad_Z, slice_x), 1, true);

      d_weights[i * k_size + j] += res;
    }
  }

  d_x = torch::slice(d_x, 2, 2, H + 2);
  d_x = torch::slice(d_x, 3, 2, W + 2);
  return {d_x, d_weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &horder_forward, "High Order forward");
  m.def("backward", &horder_backward, "High Order backward");
}
