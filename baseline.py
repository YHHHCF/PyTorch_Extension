import torch
import torch.nn as nn

class BaselineModule(nn.Module):
    def __init__(self):
        print("init")
        pass
    def forward(self, v):
        return v;

if __name__ == "__main__":
    baseline = BaselineModule();
