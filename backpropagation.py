import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)


# Forward pass: compute predicted loss
y_pred = w * x
loss = (y_pred - y) ** 2
print("Loss before backward pass:", loss)

# Backward pass: compute gradients
loss.backward()
print("Gradient of w:", w.grad)
