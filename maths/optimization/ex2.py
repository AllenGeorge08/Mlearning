# Momentum comparison. Run SGD with momentum values [0.0, 0.5, 0.9, 0.99] on the Rosenbrock function. Track the loss at every step. Which momentum value converges fastest? Which overshoots?
import matplotlib.pyplot as plt 

# OBSERVED = More momentum can make convergence faster, but too much momentum can cause large oscillations/overshooting.
# Momentum 0.99 convergess fastest means, it gets the parameters (x,y) toward (1,1) causing the loss to approach 0, faster than the other momentum values
def rosenbrock(params):  #loss
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]


class SGDMomentum:  #momentum
    def __init__(self,lr=0.001,momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None 

    #accumulate past gradients into a velocity term,help accelerate conssisten movement and dampen oscillations
    def step(self,params,grads):
        if self.velocity is None:
            self.velocity =[0.0]*len(params)
        self.velocity = [self.momentum*v + g for v,g in zip(self.velocity,grads)]
        return [p-self.lr*v for p,v in zip(params,self.velocity)]



momentum_values = [0.0, 0.5, 0.9, 0.99]

all_losses = {}

# momentum = 0.99 means the velocity is 99% of it's previous value..so imagine the optimizer moving downhill.
# the optimizer moving downhill ... 
# gradient-> velocity -> velocity -> velocity..
for momentum in momentum_values:
    params = [-1.5,2.0]
    momentum_optimizer = SGDMomentum(lr=0.001,momentum=momentum)
    losses = []
    
    for step in range(5000):
        gradient = rosenbrock_gradient(params)
        params = momentum_optimizer.step(params,gradient)
        loss = rosenbrock(params)
        losses.append(loss)
        print(f"Loss at momentum {momentum} | -> Final Loss={losses[-1]:.6f}")

    all_losses[momentum] = losses 


for momentum,losses in all_losses.items():
    plt.plot(losses,label=f"momentum={momentum}")

    
plt.xlabel('Step')
plt.ylabel("Loss")
plt.title("Momentum Comparison on Rosenbrock")
# plt.yscale("log")
plt.legend()
plt.savefig('sgdplot.png')




