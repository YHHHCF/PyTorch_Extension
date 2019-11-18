import os
import numpy as np
import torch
import torch.nn as nn
from dataloader import *

class SecondOrderConv(nn.Module):
    def __init__(self):
        super(SecondOrderConv, self).__init__()

    def forward(self, x, filter):
        B, _, H, W = x.shape  # (B, C, H, W)
        filter = filter.reshape(B, -1, 1, H, W).transpose(0,1)  # (kernel_size^2, B, 1, H, W)
        Z = torch.zeros_like(x)  # (B, C, H, W)
        k = 0
        x = torch.nn.ConstantPad2d((2, 2, 2, 2), 0)(x)  # (B, C, H+4, W+4)
        
        for hh in range(5):
            for ww in range(5):
                X_c = x[:, :, hh : H + hh, ww : W + ww] # (B, C, H, W)
                Z += X_c * filter[k] # (B, 1, H, W) * (B, C, H, W)
                k += 1

        return Z

class BaselineModule(nn.Module):
    def __init__(self):
        super(BaselineModule, self).__init__()
        self.func = SecondOrderConv()
        self.conv = nn.Conv2d(in_channels=3, out_channels=25, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        res = x
        filter = self.conv(x)
        x = self.func(x, filter)
        x += res
        return x

def run(model, loader):
    for bid, (imgs, _) in enumerate(loader):
        print(bid)
        out = model(imgs)
        loss = torch.sum(out)
        print("loss: ", loss)
        loss.backward()

if __name__ == "__main__":
    path = "./data/img_dict.npy"
    b_size = 64
    n_workers = 8

    baseline = BaselineModule()
    dataset = ImageDataset(path)
    loader = get_loader(dataset, b_size, n_workers)
    run(baseline, loader)
