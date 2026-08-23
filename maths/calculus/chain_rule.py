# z = wx+ b, alpha =  sigma(z) and  L = l(a)
#dL/dw = dL/da*da/dZ*dz/dW

#Sample implementation

from ctypes import pointer
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivate(a):
    return a*(1-a)

def linear_layer(x,w,b):
    return w*x+b 


x = 2.0 
w = 0.5 
b = 1.0 


z = linear_layer(x,w,b)
w = 0.5 
b = 1.0

a = sigmoid(z)

print("Linear Layer : ",z)
print("Sigmoid: ",a)


dl_da = 2.0 
da_dz = sigmoid_derivate(a)
dz_dw = x 

dL_dw = dl_da*da_dz*dz_dw

# f(x,y) = x**2+y**2 , Del*F = [2x 2y]
# H = [2 0
       #0 2]

       
def numerical_gradient(f,point,h=1e-7):
    gradient = []

    for i in  range(len(point)):
        point_plus = point.copy()
        point_minus =point.copy()

        point_plus[i] += h
        point_minus[i] -= h 

        partial = (
            f(point_plus) - f(point_minus)
        )/(2*h)

        gradient.append(partial)

    return gradient 



def numerical_hessian(f,point,h=1e-5):
    n = len(point)
    H = np.zeros((n,n))

    for i in range(n):
        point_plus = point.copy()
        point_minus = point.copy()

        point_plus[i] += h 
        point_minus[i] -= h 

        grad_plus = np.array(
            numerical_gradient(f,point_plus)
        )

        grad_minus = np.array(
            numerical_gradient(f,point_minus)
        )

        H[:,i] = (grad_plus-grad_minus)/(2*h)

    
    return H


def f(point):
    x,y = point 
    return x**2 + y**2 

H = numerical_hessian(
    f,
    np.array([1.0,1.0])
)

print(H)


eigenvalues = np.linalg.eigvals(H)

print(eigenvalues)