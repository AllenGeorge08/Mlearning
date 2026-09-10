import numpy as np 

A = np.array([
    [1,2],
    [3,4]
])

B = np.array([
    [10,20],
    [30,40]
])

print("-"*50)
print(A+B)
print("-"*50)
print(A*B)
print("-"*50)
print(A.sum(axis=0))
print("-"*50)
print(A.sum(axis=1))
print("-"*50)
print(A.mean(axis=0))
print("-"*50)
print(A.mean(axis=1))


# Broadcasting
X = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

bias = np.array([10,20,30])
result = X+bias 

print(result)
print(result.shape)