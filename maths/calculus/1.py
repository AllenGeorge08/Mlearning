import numpy as np

def numerical_derivative(f,x,h=1e-7):
    return (f(x+h)-f(x-h))/(2*h)


def f(x):
    return x**2


print(numerical_derivative(f,3))


def analytical_derivate(x):
    return 2*x 


print(numerical_derivative(f,3))
print(analytical_derivate(3))


def f2(point):
    x,y = point 
    return x**2+3*x*y+y**2 


def df_dx(x,y):
    return 2*x+3*y 


def df_dy(x,y):
    return 2*x+3*y


def numerical_gradient(f,point,h=1e-7):
    gradient = []

    for i in  range(len(point)):
        point_plus = point.copy()
        point_minus =point.copy()

        point_plus[i] += h
        point_plus[i] -= h 

        partial = (
            f(point_plus) - f(point_minus)
        )/(2*h)

        gradient.append(partial)

    return gradient 


gradient = numerical_gradient(
    f2,
    [1.0,2.0]
)
print(gradient)


# x = x-learning_rate*gradient
def grad_descent(f):
    x = 5.0
    learning_rate = 0.1

    for step in range(20):
        gradient = 2*x 
        x = x-learning_rate*gradient 

        print(step,x,f)


# print("Gradient descent of f(5) for f(x) = x**2")
# grad_descent(f(5))


def f_2d(point):
    x,y = point 
    return x**2 + y**2

# if f(x,y) = x**2+y**2, it's min is (0,0) and it's gradient is: Delta-f = [2x 2y] , start at (4,3)
def grad_desc_2d():
    point =  np.array([4.0,3.0])
    learning_rate=0.1

    for step in range(30):
        gradient = np.array([
            2*point[0],
            2*point[1]
        ])

        point = point - learning_rate*gradient 
        print(step,point,f_2d(point))






