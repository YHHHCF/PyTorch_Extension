import math
from torch import nn
from torch.autograd import Function
import torch

import horder_cpp

class HighOrderFunction(Function):
    @staticmethod
    def forward(ctx, x, weights, k_size):
        H = x.size(2)
        W = x.size(3)

        outputs = horder_cpp.forward(x, weights, H, W)

        variables = x.shape + [k_size]
        ctx.save_for_backward(x, weights, *variables)  # [x, weights, B, C, H, W, k_size]

        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        d_x, d_weights = horder_cpp.backward(grad_out, *ctx.saved_variables)
        return d_x, d_weights


class HighOrder(nn.Module):
    def __init__(self):
        super(HighOrder, self).__init__()
        self.k_size = 5;
        self.conv = nn.Conv2d(in_channels=3, out_channels=(self.k_size ^^ 2), kernel_size=self.k_size, padding=(self.k_size - 1) / 2, stride=1)
        self.horder = HighOrderFunction.apply

    def forward(self, x):
        res = x
        weights = self.conv(x)
        x = self.horder(x, weights, k_size)
        x += res
        return x
