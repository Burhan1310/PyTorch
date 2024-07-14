import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss  # / float(predicted.shape[0])

# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 1: [0 1 0]
Y = np.array([1, 0, 0])

#y_pred has probabilities
Y_pred_good = ([0.7, 0.2, 0.1])
Y_pred_bad = ([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

# in PyTorch
loss = nn.CrossEntropyLoss()

Y1 = torch.tensor([0])
# n_samples x n_classes = 0x3
Y_pred_good1 = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad1 = torch.tensor([[0.5, 2.0, 0.3]])

l11 = loss(Y_pred_good1, Y1)
l22 = loss(Y_pred_bad1, Y1)

print(l11.item())
print(l22.item())

_, predictions1 = torch.max(Y_pred_good1, 1)
_, predictions2 = torch.max(Y_pred_bad1, 1)

print(predictions1)
print(predictions2)


# for multiple samples
# 3 samples
Y1 = torch.tensor([2, 0, 1])

# n_samples x n_classes = 0x3
Y_pred_good1 = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad1 = torch.tensor([[2.1, 1.0, 2.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l11 = loss(Y_pred_good1, Y1)
l22 = loss(Y_pred_bad1, Y1)

print(l11.item())
print(l22.item())

_, predictions1 = torch.max(Y_pred_good1, 1)
_, predictions2 = torch.max(Y_pred_bad1, 1)

print(predictions1)
print(predictions2)