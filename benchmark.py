import argparse
import math
import time

import torch

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])

options = parser.parse_args()

if options.example == 'py':
    from python.horder import HighOrder
elif options.example == 'cpp':
    from cpp.horder import HighOrder
else:
    from cuda.horder import HighOrder
    options.cuda = True

# device = torch.device("cuda")
device = torch.device("cuda")
dtype = torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}

# path for real data
path = "../data/img_dict.npy"
batch_size = 2
num_workers = 1
runs = 10
scale_name = 'ms'

# fake(random initialized data)
X = torch.randn(batch_size, 3, 224, 224).to(device)
model = HighOrder().to(device, dtype)

# force initialization
output = model(X)
output.sum().backward()

forward_min = math.inf
forward_time = 0
backward_min = math.inf
backward_time = 0

for _ in range(runs):
    model.zero_grad()

    start = time.time()
    output = model(X)
    elapsed = time.time() - start
    forward_min = min(forward_min, elapsed)
    forward_time += elapsed

    start = time.time()
    output.sum().backward()
    elapsed = time.time() - start
    backward_min = min(backward_min, elapsed)
    backward_time += elapsed

scale = TIME_SCALES[scale_name]
forward_min *= scale
backward_min *= scale
forward_average = forward_time / runs * scale
backward_average = backward_time / runs * scale

print('Forward: {0:.1f}/{1:.1f} {4} | Backward {2:.1f}/{3:.1f} {4}'.format(
    forward_min, forward_average, backward_min, backward_average,
    scale_name))

