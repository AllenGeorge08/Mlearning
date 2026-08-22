import numpy as np

def mat_vec_mul(matrix,vector):
    return [
        #matrix[i][j] * vector[j]
        sum(matrix[i][j]*vector[j] for j in range(len(vector))) for i in range(len(matrix))
    ]


A = np.array([
    [2,1],
    [3,4]
])

V = [4,2]

print(mat_vec_mul(A,V))

def mat_mul(A,B):
    rows_a  = len(A)
    cols_a = len(A[0])
    cols_b = len(B[0])

    return [
        [
            sum(A[i][k]*B[k][j] for k in range(cols_a)) for j in range(cols_b)
        ]
        for i in range(rows_a)
    ]

j = np.array([
    [22,33],
    [44,55]
])

print(mat_mul(A,j))

