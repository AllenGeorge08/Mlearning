heads = 7
tails = 3
total = heads+tails 

mle = heads/total 
print(f"MLE :{mle}")

alpha,beta = 2,2

map_estimate = (heads+alpha-1)/(heads+tails+beta+alpha-2)
print(f"Map Estimate : {map_estimate}")
