import numpy as np

def numerical_gradient_2d(f,x,y,h=1e-5):
    df_dx = (f(x+h,y)-f(x-h,y))/(2*h)
    df_dy = (f(x,y+h)-f(x,y-h))/(2*h)

    return df_dx,df_dy

def f(x,y):
    return (x-3)**2 + (y+1)**2

def grad_descent(f,point,steps=200):
    learning_rate = 0.01 
    x,y = point 

    for step in range(steps):
        df_dx,df_dy = numerical_gradient_2d(f,x,y)
        x = x-learning_rate*df_dx
        y = y-learning_rate*df_dy

        print(step,x,y)

    if (step+1)%10==0:
        print(f"Step {step+1:3d} | x: {x:.4f}, y: {y:.4f}")

    return (x,y)


final_point = grad_descent(f,(0,0))
print(f"Final Point: {final_point}")

