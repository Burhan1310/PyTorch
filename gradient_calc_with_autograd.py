import torch

x = torch.rand(3, requires_grad=True)
print(x)

'''y = x+2
print(y)

z = y*y*2
z = z.mean()
print(z)

z.backward() #dz/dx
print(x.grad)
'''
#prevent for tracking the history
#x.rquires_grad_(False)
'''x.requires_grad_(False)
print(x)

#x.detach()
y = x.detach()
print(y)

#with torch.no_grad():
with torch.no_grad():
    y = x + 2
    print(y)
'''
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights * 3).sum()

    model_output.backward()

    print(weights.grad)
     #to prevent this update
    weights.grad.zero_()
