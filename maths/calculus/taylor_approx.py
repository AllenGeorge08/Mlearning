import numpy as np


# Suppose f(x,y) = [x**2+y xy]
# Calculate J = [2x 1
                # y x]
def numerical_jacobian(f,point,h=1e-5):
    point = np.array(point,dtype=float)

    output = np.array(f(point))

    m = len(output)
    n = len(point)

    J = np.zeros((m,n))

    for j in range(n):
        plus = point.copy()
        minus = point.copy()

        plus[j] += h 
        minus[j] -= h 

        J[:,j] = (np.array(f(plus) - np.array(f(minus))))/(2*h)

    return J 

def f(point):
    x,y = point 
    return np.array([
        x**2+y,x*y
    ])


print(numerical_jacobian(f,[2.0,3.0]))