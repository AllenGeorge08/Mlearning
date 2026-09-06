import random 

data = list(range(1,101))

# Batch GD
batch = data 
gradient = sum(batch)/len(batch)
print(gradient)


# SGD
sample = random.choice(data)
gradient = sample 
print(gradient)

# y' = wx 
# L = (y'-y)**2
# L =(wx-y)**2
# dL/dw = 2(wx-y)x
def mini_batch_gradient(w,batch):
    total_gradient = 0 

    for x,y in batch:
        prediction = w*x 
        gradient  = 2*(prediction-y)*x 
        total_gradient += gradient  # -2 + -8+ -18 

    return total_gradient/len(batch) #-28/3 = -9.33 -> optimizer -> w = w-lr*gradient
    # lr = 0.1,  1 = 1-0.1*-9.333 = 1.933 =  answer


batch = [
    (1,2),
    (2,4),
    (3,6)
]

w = 1 

print(mini_batch_gradient(1,batch))