import math
from torch import nn
from torch.autograd import Function
import torch

import horder_cuda

class HighOrderFunction(Function):
    @staticmethod
    def forward(ctx, x, weights):
        k_size = 5
        x = x.contiguous()
        weights = weights.contiguous()

        outputs = horder_cuda.forward(x, weights, k_size)
        ctx.save_for_backward(x, weights, torch.tensor(k_size))
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        d_x, d_weights = horder_cuda.backward(grad_out.contiguous(), *ctx.saved_variables)
        return d_x, d_weights


class HighOrder(nn.Module):
    def __init__(self):
        super(HighOrder, self).__init__()
        k_size = 5;
        self.conv = nn.Conv2d(in_channels=3, out_channels=(k_size ** 2), kernel_size=k_size, padding=int((k_size - 1) / 2), stride=1)
        self.horder = HighOrderFunction.apply

    def forward(self, x):
        res = x
        weights = self.conv(x)
        B, _, H, W = x.shape  # (B, C, H, W)
        weights = weights.reshape(B, -1, 1, H, W).transpose(0,1)  # (kernel_size^2, B, 1, H, W)
        x = self.horder(x, weights)
        x += res
        return x
