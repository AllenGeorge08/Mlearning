#  Vanilla gradient descent on the rosenbrock functions
# Learning rates =  [0.0001, 0.0005, 0.001, 0.005, 0.01]


import math


def rosenbrock(params):
    x,y = params 
    return (1-x)**2 + 100*(y-x**2)**2 


def rosenbrock_gradient(params):
    x,y = params 
    df_dx = -2*(1-x)+ 200*(y-x**2)*(-2*x)
    df_dy = 200*(y-x**2)
    return [df_dx,df_dy]


class GradientDescent:
    def __init__(self,lr):
        self.lr = lr 

    def step(self,params,grads):
        return [p-self.lr*g for p,g in zip(params,grads)]
    


x = -1.5
y = 2.0 


learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]

# With a large learning rate -> large gradient * large lr -> Huge update -> oversshoot -> even larger gradient -> even bigger update
# x or y becomes astronomically large and python cannot contain them

# Overflow error is evidence tha t0.005 diverged ..before that all converged.
# | Learning rate | Final loss | Result                 |
# | ------------: | ---------: | ---------------------- |
# |        0.0001 |   4.484731 | Converging, but slowly |
# |        0.0005 |   0.091404 | Converging             |
# |         0.001 |   0.006514 | Converging             |
# |         0.005 |   Overflow | Diverged               |
# |          0.01 |   Overflow | Diverged               |

# Largest learning rate that converges = 0.001...
for lr in learning_rates:
    try:
        
        params = [-1.5,2.0] 

        diverged = False

        optimizer = GradientDescent(lr=lr)

        for step in range(5000):
            grads = rosenbrock_gradient(params)
            params = optimizer.step(params,grads)


            if not all(math.isfinite(p) for p in params):
                diverged=True
                break 
        if diverged:
            print(f"lr={lr} -> DIVERGED")
        else:
            loss = rosenbrock(params)
            print(f"lr={lr} -> final loss = {loss:.6f}")
     
    except OverflowError as e:
        print(f"Error = {e}")
        print(f"Largest Learning Rate that converges = {lr}")