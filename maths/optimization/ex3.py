# Saddle point escape. Define the function f(x, y) = x^2 - y^2 (a saddle point at the origin). Start at (0.01, 0.01). Compare how vanilla GD, SGD with momentum, and Adam behave. Which escapes the saddle point?
from ex2 import SGDMomentum
from ex1 import GradientDescent

# Imagine we're training a NN and you reach a saddle point(0,0)
# If your optimizer somehow gets stuck there..(no meaningful update)
# But there're dirctions where the losss can decrease...
# So we want the optimizer to keep moving toward lower-loss regions rather than getting stuck around a saddle.
# When gradient reaches 0, but more loss is there..we gotta do it...


# Adam >> sgd and vanilla descent

# | Optimizer  |        x |            y |
# | ---------- | -------: | -----------: |
# | Vanilla GD | 0.001326 |     0.072446 |
# | Momentum   | 0.001022 |     0.049346 |
# | Adam       | 0.000029 | **0.142863** |


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]

        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]


def func(x,y):
    return x**2-y**2

def f_grad(x,y):
    return (2*x,-2*y)

params  = [0.01,0.01]

optimizer = Adam()
print('Adam Optimizer ....')
for step in range(100):
    grads = f_grad(*params)
    params = optimizer.step(params,grads=grads)
    print(f"Step = {step}, Params = {params}")
    

# sgd_optimizer = SGDMomentum()
# print("SGD Momemnutm Optimizer...")
# for step in range(100):
#     grads = f_grad(*params)
#     params = sgd_optimizer.step(params,grads=grads)
#     print(f"Step = {step},Params = {params}")

# vanilla_optimizer = GradientDescent(0.01)
# for step in range(100):
#     grads = f_grad(*params)
#     params = vanilla_optimizer.step(params,grads)
#     print(f"Step = {step} ->",f"Params = {params}")
