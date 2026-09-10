import numpy as np 
import torch

A = np.array([
    [1,2,3],
    [4,5,6]
])

print(A.shape)
T = A.T 
print(T)

x = torch.tensor([
    [1,2,3],
    [4,5,6]
])

print(x.is_contiguous())  #natural order of elementss in memory
y = x.transpose(0,1)
print(y.is_contiguous())