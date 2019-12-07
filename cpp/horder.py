import math
from torch import nn
from torch.autograd import Function
import torch

import horder_cpp

class HighOrderFunction(Function):
    @staticmethod
    def forward(ctx, x, weights):
        H = x.size(2)
        W = x.size(3)

        outputs = horder_cpp.forward(x, weights, H, W)
        ctx.save_for_backward(x, weights, torch.tensor(5))

        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        d_x, d_weights = horder_cpp.backward(grad_out, *ctx.saved_variables)
        return d_x, d_weights


class HighOrder(nn.Module):
    def __init__(self):
        super(HighOrder, self).__init__()
        self.k_size = 5;
        self.conv = nn.Conv2d(in_channels=3, out_channels=(self.k_size ** 2), kernel_size=self.k_size, padding=int((self.k_size - 1) / 2), stride=1)
        self.horder = HighOrderFunction.apply

    def forward(self, x):
        res = x
        weights = self.conv(x)
        B, _, H, W = x.shape  # (B, C, H, W)
        weights = weights.reshape(B, -1, 1, H, W).transpose(0,1)  # (kernel_size^2, B, 1, H, W)
        x = self.horder(x, weights)
        x += res
        return x
