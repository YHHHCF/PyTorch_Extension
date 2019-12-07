import os
import sys 
sys.path.append("..")
import time
import numpy as np
import torch
import torch.nn as nn
from tools.dataloader import *

class HighOrderFunction(nn.Module):
    def __init__(self):
        super(HighOrderFunction, self).__init__()

    def forward(self, x, weights):
        B, _, H, W = x.shape  # (B, C, H, W)
        Z = torch.zeros_like(x)  # (B, C, H, W)
        k = 0
        x = torch.nn.ConstantPad2d((2, 2, 2, 2), 0)(x)  # (B, C, H+4, W+4)
        
        for hh in range(5):
            for ww in range(5):
                X_c = x[:, :, hh : H + hh, ww : W + ww] # (B, C, H, W)
                Z += X_c * weights[k] # (B, 1, H, W) * (B, C, H, W)
                k += 1

        return Z

class HighOrder(nn.Module):
    def __init__(self):
        super(HighOrder, self).__init__()
        self.horder = HighOrderFunction()
        self.conv = nn.Conv2d(in_channels=3, out_channels=25, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        res = x
        weights = self.conv(x)

        B, _, H, W = x.shape  # (B, C, H, W)
        weights = weights.reshape(B, -1, 1, H, W).transpose(0,1)  # (kernel_size^2, B, 1, H, W)

        x = self.horder(x, weights)
        x += res
        return x

# run the module using real image and data loader
def loader_run(model, path, batch_size, num_workers, rounds):
    dataset = ImageDataset(path)
    loader = get_loader(dataset, batch_size, num_workers)

    forward = 0
    backward = 0
    idx = 0

    while(idx < rounds):
        for _, (imgs, _) in enumerate(loader):
            start = time.time()
            out = model(imgs)
            forward += time.time() - start

            start = time.time()
            out.sum().backward()
            backward += time.time() - start

            idx += 1
            if (idx >= rounds):
                break

    print('Forward: {:.2f} ms | Backward {:.2f} ms'.format(forward * 1e3/rounds, backward * 1e3/rounds))

# run the module with fake tensor directly
def no_loader_run(model, batch_size, rounds):
    x = torch.randn(batch_size, 3, 224, 224)

    forward = 0
    backward = 0

    for _ in range(rounds):
        start = time.time()
        out = baseline(x)
        forward += time.time() - start

        start = time.time()
        out.sum().backward()
        backward += time.time() - start

    print('Forward: {:.2f} ms | Backward {:.2f} ms'.format(forward * 1e3/rounds, backward * 1e3/rounds))


if __name__ == "__main__":
    path = "../data/img_dict.npy"
    batch_size = 1
    num_workers = 1
    rounds = 3

    baseline = HighOrder()

    # run with data loader and batched real data
    loader_run(baseline, path, batch_size, num_workers, rounds)

    # run with random data
    no_loader_run(baseline, batch_size, rounds)
