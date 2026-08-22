# Z=WX+B

Z= [[1,2,3]
,[4,5,6],
[7,8,9]]

W = [[10,11,12],
[13,14,15],
[16,17,18]]

rows_a = len(Z)
cols_a = len(Z[0])
rows_b = len(W)
cols_b = len(W[0])

C =  [[0]* cols_b for _ in range(len(Z))]

if cols_a != cols_b:
    raise ValueError("Matrices cannot be multiplied")

for i in range(len(Z)):
    for j in range(len(W)):
        for k in range(cols_a):
            C[i][j] += Z[i][k]*W[k][j]


print(C)

bias = [1,2,3]

for i in range(len(C)):
    for j in range(len(bias)):
        C[i][j] += bias[j]
    

print(C)