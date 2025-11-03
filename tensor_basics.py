import torch

x = torch.empty(3)
y = torch.empty(2, 4)
print(x) # This will print an uninitialized tensor of size 3
print(y) # This will print an uninitialized tensor of size (2, 4)

z = torch.rand(2, 3)
print(z) # This will print a tensor of size (2, 3) with random

a = torch.zeros(2, 3)
print(a) # This will print a tensor of size (2, 3) filled with zeros

b = torch.ones(2, 3)
print(b) # This will print a tensor of size (2, 3) filled with ones