import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

class BaselineModule(nn.Module):
    def __init__(self):
        super(BaselineModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)

    def forward(self, v):
        print(v.shape)
        v = self.conv(v)
        return v

def get_loader():
    dataset = torchvision.datasets.ImageFolder(
        root='.',
        transform=torchvision.transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=True
    )
    return loader


def run(model, loader):
    for bid, (imgs, _) in enumerate(loader):
        print(bid)
        out = model(imgs)
        loss = torch.sum(out)
        print("loss: ", loss)
        loss.backward()

if __name__ == "__main__":
    baseline = BaselineModule()
    loader = get_loader()
    run(baseline, loader)

