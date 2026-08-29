heads = 7
tails = 3
total = heads+tails 

mle = heads/total 
print(f"MLE: {mle}")

alpha,beta = 2,2


map_estimate = (heads+alpha-1)/(heads+tails+alpha+beta-2)
# The prior pulled our estimate slightly toward 0.5. compared to mle i.e 0.7
print(f"MAP: {map_estimate}")



priors = [
    (1, 1),
    (2, 2),
    (10, 10),
    (100, 100)
]

# We want a prior around 0.5 , thus stronger prior = more pull towards preffered value
for alpha,beta in priors:
    map_estimate = (heads+alpha-1)/(heads+tails+alpha+beta-2)
    print(f"Beta : {({alpha},{beta})} -> MAP: {map_estimate:.4f}")






