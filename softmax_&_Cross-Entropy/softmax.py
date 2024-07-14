import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(outputs)

X = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(X, dim=0)
print(outputs)