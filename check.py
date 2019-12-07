import torch
from python.horder import HighOrderFunction as ho1
from cpp.horder import HighOrderFunction as ho2


def check_equals(v1, v2):
    if (v1.shape != v2.shape):
        print("not equals")
        return

    abs_diff = (v1 - v2).abs()
    if (abs_diff.sum() > 1e-1):
        print("not equals")
        return

    print("equals")
    return

# create input parameters(different instances) with same values
x1 = torch.randn((1, 3, 224, 224), requires_grad = True)
weights1 = torch.randn((25, 1, 1, 224, 224), requires_grad = True)

x2 = x1.clone().detach().requires_grad_(True)
weights2 = weights1.clone().detach().requires_grad_(True)
# create python and cpp models
model1 = ho1()
model2 = ho2.apply

# check forward pass
out1 = model1(x1, weights1)
out2 = model2(x2, weights2)

check_equals(out1, out2)

# check backward pass
out1.sum().backward()
out2.sum().backward()

check_equals(x1.grad, x2.grad)
check_equals(weights1.grad, weights2.grad)
