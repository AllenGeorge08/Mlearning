import numpy as np

def eigenvalue(matrix):
    a,b = matrix[0]
    c,d = matrix[1]

    trace = a+d
    determinant = a*d - b*c
    discriminant_eqn = (-trace**2) - 4(1)(determinant)  #b**2 - 4ac where a=1  , skipped forming the char eqn
    sqrt_disc = discriminant_eqn**0.5
    lambda1 = (trace+sqrt_disc)/2
    lambda2 = (trace-sqrt_disc)/2
    return (lambda1,lambda2)