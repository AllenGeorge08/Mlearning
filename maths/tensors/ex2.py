# # Broadcasting
import torch 

# a = torch.tensor([
#     [1],
#     [2],
#     [3]
# ])
a= torch.randn(3,1)
print(a.shape)
# b = torch.tensor([
#     [10,20,30,40]
# ])

b = torch.randn(1,4)
print(b.shape)

c = a+b 

print(c)
print(c.shape)