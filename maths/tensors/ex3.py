import torch 

a = torch.tensor([1,2,3])
b= torch.tensor([4,5,6])

result = torch.einsum("i,i->",a,b)
print(result)
