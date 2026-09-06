# Implement learning rate decay. Add an exponential decay schedule to the GradientDescent class: lr = lr_0 * 0.999^step. Compare convergence with and without decay on the Rosenbrock function.

# Optimizer -> takes ssteps of nearly same size , lr=0.001 always

# With decay, learning rate smaller over time 

# step 0     lr = 0.001
# step 100   lr ≈ 0.000904
# step 1000  lr ≈ 0.000368
# step 5000  lr ≈ 0.0000067

def rosenbrock(params):
    x,y = params 
    return (1-x)**2 + 100*(y-x**2)**2 


def rosenbrock_gradient(params):
    x,y = params 
    df_dx = -2*(1-x)+ 200*(y-x**2)*(-2*x)
    df_dy = 200*(y-x**2)
    return [df_dx,df_dy]



class GradientDescent:
    def __init__(self,lr=0.001,decay=False):
        self.lr_0 = lr 
        self.decay = decay 
    

    def step(self,params,grads,step):
        if self.decay:
            lr = self.lr_0 * (0.999**step)
        else:
            lr = self.lr_0

        return [p-lr*g for p,g in zip(params,grads)]



learning_rate = 0.001

params = [-1.5,2.0]
optimizer  =GradientDescent(lr=learning_rate,decay=False)

for step in range(5000):
    grads = rosenbrock_gradient(params)
    params = optimizer.step(params,grads,step)


print("Without decay: ")
print("Params: ",params)
print("Loss: ",rosenbrock(params))


optimizer  =GradientDescent(lr=learning_rate,decay=True)

for step in range(5000):
    grads = rosenbrock_gradient(params)
    params = optimizer.step(params,grads,step)

# Move relatively quickly at the beginning
# Take smaller and more precise steps later

print("-"*50)
print("With decay: ")
print("Params: ",params)
print("Loss: ",rosenbrock(params))

