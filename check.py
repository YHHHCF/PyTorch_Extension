import torch
from python.horder import HighOrderFunction as ho1
from cpp.horder import HighOrderFunction as ho2
from cuda.horder import HighOrderFunction as ho3

def check_equals(v1, v2):
    if (v1.shape != v2.shape):
        print("shape not equals")
        return False

    abs_diff = (v1 - v2).abs()
    abs_diff = (abs_diff > 1e-3)

    if (abs_diff.sum() > 0):
        print("value not equals")
        return False
    return True

device = torch.device("cuda")

# create input parameters(different instances) with same values
torch.manual_seed(117)
x1 = torch.randn((1, 3, 224, 224), requires_grad = True).to(device)
weights1 = torch.randn((25, 1, 1, 224, 224), requires_grad = True).to(device)

torch.manual_seed(117)
x2 = torch.randn((1, 3, 224, 224), requires_grad = True).to(device)
weights2 = torch.randn((25, 1, 1, 224, 224), requires_grad = True).to(device)

torch.manual_seed(117)
x3 = torch.randn((1, 3, 224, 224), requires_grad = True).to(device)
weights3 = torch.randn((25, 1, 1, 224, 224), requires_grad = True).to(device)

assert torch.all(torch.eq(x1, x2))
assert torch.all(torch.eq(x1, x3))
assert torch.all(torch.eq(weights1, weights2))
assert torch.all(torch.eq(weights1, weights3))

assert (id(x1) != id(x2))
assert (id(x1) != id(x3))
assert (id(x2) != id(x3))
assert (id(weights1) != id(weights2))
assert (id(weights1) != id(weights3))
assert (id(weights2) != id(weights3))

# create python and cpp models
model1 = ho1()
model2 = ho2.apply
model3 = ho3.apply

# check forward pass

out1 = model1(x1, weights2)
out2 = model2(x2, weights2)
out3 = model3(x3, weights3)

print("Python vs Cpp, forward:", check_equals(out1, out2))
print("Python vs Cuda, forward:", check_equals(out1, out3))


# check backward pass
out1.sum().backward()
out2.sum().backward()
out3.sum().backward()

print(x1.grad)
print(x2.grad)
print(x3.grad)

print(weights1.grad)
print(weights2.grad)
print(weights3.grad)

