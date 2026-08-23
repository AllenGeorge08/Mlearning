# Implement numerical_second_derivative(f, x) using numerical_derivative called twice. Verify that the second derivative of x^3 at x=2 is 12.

# $$\text{Derivative} \approx \frac{\overbrace{f(x + h) - f(x - h)}^{\text{Change in } Y}}{\underbrace{(x + h) - (x - h)}_{\text{Change in } X = 2h}} = \frac{f(x + h) - f(x - h)}{2h}$$

def numerical_derivate(f,x,h=1e-5):
    result =  (f(x+h)-f(x-h))/(2*h)
    return result 


def numerical_second_derivative_two(f,x):
    derivative=  lambda t:  numerical_derivate(f,t)
    return numerical_derivate(derivative,x)

x=2

def f(x):
    return x**3

print(numerical_second_derivative_two(f,x))