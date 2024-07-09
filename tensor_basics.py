import torch
import numpy as np


x = torch.empty(2, 3)
print(x)

y = torch.ones(2,2)
print(y.dtype)
print(x.size())

a = torch.rand(2,2)
b = torch.rand(2,2)

#Basic Operation
'''print(a)
print(b)
c = a + b
c = torch.add(a,b)
print(c)
b.add_(a)
print(b)'''


'''c = torch.sub(a,b)
c = torch.mul(a,b)
c = torch.div(a,b)
'''
#Slicing Methods
'''
d = torch.rand(5,3)
print(d)
'print(d[:,0])'
print(d[1,:])
print(d[1,1].item())
'''
#Reshaping Methods
'''e = torch.rand(4,4)
print(e)
f = e.view(16)
print(f)
f = e.view(-1, 8)
print(f)'''

#Converting numpy to torch & vice-versa
a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b))
print(b)

a.add_(1)
print(a)
print(b)

c = np.ones(5)
print(c)
d = torch.from_numpy(c)
print(d)

