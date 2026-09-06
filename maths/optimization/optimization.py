def loss(w):
    return w**2 

def gradient(w):
    return 2*w 


w = 3.0 
lr = 0.1 


# for step in range(48):
#     if abs(loss(w)) < 1e-6:
#         print(f"Converged to zero at step :{step}")
#         break

#     grad = gradient(w)
#     w = w - lr*grad 
#     print(f"Step = {step:2d},",f"W={w:.4f}",f"loss={loss(w):.4f}")
        
       

def gradient_descent(params,grads,lr):
    new_params = []

    for p,g in zip(params,grads):
        p = p-lr*g 
        new_params.append(p)
    
    return new_params


params = [3.0,2.0]
grads = [6.0,4.0]

answer = gradient_descent(params,grads,0.1)
print(f"Params : {answer}")

