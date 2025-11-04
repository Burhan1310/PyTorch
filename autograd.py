import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 2
print(z)
z = z.mean()
print(z)

z.backward() # compute gradients dz/dx
print(x.grad) # print gradients d(z)/d(x)

# Preventing tracking of history
print("\n--- Preventing tracking of history ---")
x.requires_grad_(False)
print(x)

x.detach()
print(x)

with torch.no_grad():
    y = x + 2
    print(y)


weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights * 3).sum()
    
    model_output.backward()
    print(weights.grad)
    
    weights.grad.zero_()
    
    
