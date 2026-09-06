class Momentum:
    def __init__(self,lr=0.1,beta=0.9):
        self.lr = 0 
        self.beta = beta 
        self.velocity=0

    
    def step(self,w,gradient):
        self.velocity = (self.beta*self.velocity + gradient)
        w = w - self.lr * self.velocity
        return w 
        


w = 1.0 
gradient = 0.5 

momentum_optimizer = Momentum()
ans = momentum_optimizer.step(w,gradient)
print(f"Velocity for W: {w}, gradient: {gradient} = {ans}")

