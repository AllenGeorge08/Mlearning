import numpy as np 

a = np.array([1,2,3])
b = np.array([4,5,6])

result = np.einsum("i,i->",a,b)
print(result)


result_two = np.einsum("i,j->ij",a,b)
print(result_two)

# Matrix operation C=A@B
A = np.random.rand(2,3)
B = np.random.rand(3,4)

C = np.einsum("ij,jk->ik",A,B)
print(C.shape)


# Trace
T = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

result_trace = np.einsum("ii->",T)
print(result_trace)

# Transpose
transpose = np.array([
    [1,2,3],
    [4,5,6]
])

result_transpose = np.einsum("ij->ji",transpose)
print(result_transpose)

# Batch matrix multiplication
A2 = np.random.randn(8,3,4)
A3= np.random.randn(8,4,5)

C = np.einsum("bij,bjk->bik",A2,A3)
print(C.shape)

