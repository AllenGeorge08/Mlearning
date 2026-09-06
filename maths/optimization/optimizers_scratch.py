# f(x,y)=(1−x)**2+100(y−x**2)**2

def rosebnrock_gradient(x,y):
    dx  = -2*(1-x)-400*x*(y-x**2)
    dy = 200*(y-x**2)

    return dx,dy 



# Vanilla,gradient descent
def gradient_descent(x,y,lr=0.001,steps=10000):
    for _ in range(steps):
        dx,dy = rosebnrock_gradient(x,y)
        x = x-lr*dx 
        y = y-lr*dy  
    

    return x,y  


x,y = -1.5,2.0 
(dx,dy) = rosebnrock_gradient(x,y)
print(f"Initial Gradients = {dx} {dy}")
x,y = gradient_descent(x,y)
print(f"x={x}",f"y={y}")



# Current gradient
    #   +
# Past gradients
    #   ↓
# Take smoother step
def momentum(x,y,lr=0.001,beta=0.9,steps=10000):
    vx,vy =0,0 

    for _ in range(steps):
        dx,dy = rosebnrock_gradient(x,y)
        vx = beta*vx+dx 
        by =  beta*vy + dy 

        x = x-lr*vx 
        y = y-lr*vy 

    return x,y 


print("-"*50)
x,y = -1.5,2.0 
(dx,dy) = rosebnrock_gradient(x,y)
print(f"Initial Gradients = {dx} {dy}")
x,y = momentum(x,y)
print(f"x={x}",f"y={y}")


# Adam 
# m      = gradient direction/history
# v      = gradient magnitude/history

# beta   = how much history to remember
# hat    = bias-corrected
# epsilon = prevents division by zero
def adam(x,y,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8,steps=10000):
    mx = my =0
    vx= vy=0



    for t in range(1,steps+1):
        dx,dy = rosebnrock_gradient(x,y)

        mx = beta1*mx+(1-beta1)*dx
        my = beta1*my+(1-beta1)*dy 

        vx = beta2*vx + (1-beta2)*dx**2 
        vy = beta2*vy + (1-beta2)*dy**2 

        # BiAss corrected version of mx and my, mx,my=raw gradient memory
        mx_hat = mx/(1-beta1**t)
        my_hat = my/(1-beta1**t)

        # Beta1 and Beta2 are acctually gradient histories (m and v). Controls how much history do we remember...

        # Biass corrected verssion of vx and vy
        vx_hat = vx/(1-beta2**t)
        vy_hat = vy/(1-beta2**t)

        # Epssilon isss a tniy number added to prevent div by zero..instead of divind by zero, we add 1e-8
        x -= lr*mx_hat/(vx_hat**0.5+epsilon)
        y -= lr*my_hat/(vy_hat**0.5+epsilon)

    
    return x,y



print("-"*50)
x,y = -1.5,2.0 
(dx,dy) = rosebnrock_gradient(x,y)
print(f"Initial Gradients = {dx} {dy}")
x,y = adam(x,y)
print(f"Adam -> x={x}",f"y={y}")