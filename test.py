import torch

# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("Tensor x:", x)

# Perform basic operations
y = torch.tensor([5.0, 6.0, 7.0, 8.0])
z = x + y
print("Sum of x and y:", z)

# Perform matrix multiplication
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
c = torch.matmul(a, b)
print("Matrix multiplication result:\n", c)