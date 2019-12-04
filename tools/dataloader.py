import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, path):
        imgs = np.load(path)
        self.imgs = torch.tensor(imgs, dtype=torch.float32)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        return self.imgs[index], 0

def get_loader(dataset, b_size, n_workers):
    return DataLoader(dataset, batch_size=b_size, shuffle=True, num_workers=n_workers)
