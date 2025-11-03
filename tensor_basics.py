import torch
import numpy as np

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

c = torch.randn(2, 3, dtype=torch.float64)
print(c) # This will print a tensor of size (2, 3) with values
print(c.dtype)
print(c.size())

list_tensor = torch.tensor([1, 2, 3, 4, 5])
print(list_tensor) # This will print a tensor created from the list 

# Some basic Operations
m = torch.rand(2, 2)
n = torch.rand(2, 2)
print(m)
print(n)
o = m + n
o = torch.add(m, n) # Another way to do element-wise addition
print(o) # This will print the result of element-wise addition
n.add_(m) # In-place addition
print(n) # This will print the updated tensor n after in-place addition


# Subtraction
p = m -n
p = torch.sub(m, n)
print(p)

# Multiplication
q = m * n
q = torch.mul(m, n)
print(q)

# Slicing
print("Slicing Example:")
r = torch.rand(5, 3)
print(r)
print(r[:, 1]) # This will print all rows of the second column
print(r[1, :]) # This will print all columns of the second row
print(r[1, 1]) # This will print the element at row 1, column 1s
print(r[1, 1].item()) # This will print the value as a standard Python number

# Reshaping
print("Reshaping Example:")
s = torch.rand(4, 4)
print(s)
t = s.view(16) # Reshape to 1D tensor of size 16
print(t)
t = s.view(-1, 8) # Reshape to 2D tensor with 8 columns, rows inferred
print(t)
print(t.size())

# Converting Torch Tensor to NumPy Array
print("Torch to NumPy Example:")
e = torch.ones(5)
print(e)
f = e.numpy()
print(f) # This will print the NumPy array converted from the Torch tensor
print(type(f)) # This will print <class 'numpy.ndarray'>
e.add_(1)
print(e)
print(f)

# Converting NumPy Array to Torch Tensor
print("NumPy to Torch Example:")
g = np.ones(5)
print(g)
h = torch.from_numpy(g)
print(h) # This will print the Torch tensor converted from the NumPy array