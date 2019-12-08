import torch
from python.horder import HighOrderFunction as ho1
from cpp.horder import HighOrderFunction as ho2
from cuda.horder import HighOrderFunction as ho3

def check_equals(v1, v2):
    if (v1.shape != v2.shape):
        print("shape not equals")
        return

    abs_diff = (v1 - v2).abs()

    print(v1)
    print(v2)

    if (abs_diff.sum() > 1e-1):
        print("value not equals")
        return

    print("equals")
    return

device = torch.device("cuda")

# create input parameters(different instances) with same values
x1 = torch.randn((1, 3, 224, 224), requires_grad = True).to(device)
weights1 = torch.randn((25, 1, 1, 224, 224), requires_grad = True).to(device)

x2 = x1.clone().detach().requires_grad_(True).to(device)
weights2 = weights1.clone().detach().requires_grad_(True).to(device)

x3 = x1.clone().detach().requires_grad_(True).to(device)
weights3 = weights1.clone().detach().requires_grad_(True).to(device)

# create python and cpp models
# model1 = ho1()
model2 = ho2.apply
model3 = ho3.apply

# x1.to(device)
x2.to(device)
x3.to(device)
# weights1.to(device)
weights2.to(device)
weights3.to(device)

# print("x")
# print(x3)
# print("weights")
# print(weights3)

# check forward pass
# print(id(x2))
# print(id(x3))
# check_equals(x2, x3)

# print(id(weights2))
# print(id(weights3))
# check_equals(weights2, weights3)

out2 = model2(x2, weights2)
out3 = model3(x3, weights3)

check_equals(out2, out3)

# check backward pass
out2.sum().backward()
out3.sum().backward()

check_equals(x2.grad, x3.grad)
check_equals(weights2.grad, weights3.grad)
