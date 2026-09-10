# Easy -- Reshape round-trip. Take a tensor of shape (2, 3, 4). Reshape it to (6, 4), then to (24,), then back to (2, 3, 4). Verify element order is preserved at each step by printing the flat data.
import torch

def log():
    print("-"*50)



x = torch.arange(24).reshape(2,3,4)
print(x)
# print(x.size())
print(x.flatten())
# print(f"Dim of x before reshaping :{x.dim()}")

log()
y = torch.reshape(x,(6,4))
# print(y.size())
print(y.flatten())
# print(f"Dim of x : {x.dim()} after reshaping")

log()
z = torch.reshape(x,(24,))
print(z.flatten())
print(z)

log()
a = torch.reshape(x,(2,3,4))
print(a.flatten())
print(a)

