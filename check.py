import torch
from python.horder import HighOrderFunction as ho1
from cpp.horder import HighOrderFunction as ho2
from cuda.horder import HighOrderFunction as ho3

def check_equals(v1, v2):
    if (v1.shape != v2.shape):
        return "incorrect, shape not equals"

    abs_diff = (v1 - v2).abs()

    if (torch.max(abs_diff) > 1e-3):
        return "incorrect, value not equals"

    return "correct"

device = torch.device("cuda")
side = 256
k_size = 7
c_size = 3
b_size = 16

# create input parameters(different instances) with same values
torch.manual_seed(117)
x1 = torch.randn((b_size, c_size, side, side), device=device, requires_grad=True)
weights1 = torch.randn((k_size * k_size, b_size, 1, side, side), device=device, requires_grad=True)

torch.manual_seed(117)
x2 = torch.randn((b_size, c_size, side, side), device=device, requires_grad=True)
weights2 = torch.randn((k_size * k_size, b_size, 1, side, side), device=device, requires_grad=True)

torch.manual_seed(117)
x3 = torch.randn((b_size, c_size, side, side), device=device, requires_grad=True)
weights3 = torch.randn((k_size * k_size, b_size, 1, side, side), device=device, requires_grad=True)

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
out1 = model1(x1, weights1)
out2 = model2(x2, weights2)
out3 = model3(x3, weights3)

print("Python vs Cpp, forward:", check_equals(out1, out2))
print("Python vs Cuda, forward:", check_equals(out1, out3))

# check backward pass
out1.sum().backward()
out2.sum().backward()
out3.sum().backward()

# grad is only avaible when the model and data is on CPU
print("Python vs Cpp, backward x:", check_equals(x1.grad, x2.grad))
print("Python vs Cuda, backward x:", check_equals(x1.grad, x3.grad))

print("Python vs Cpp, backward weights:", check_equals(weights1.grad, weights2.grad))
print("Python vs Cuda, backward weights:", check_equals(weights1.grad, weights3.grad))
