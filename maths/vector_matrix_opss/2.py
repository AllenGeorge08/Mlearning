import numpy as np

def adjugate(A):
    A = np.asarray(A,dtype=float)

    if A.shape != (3,3):
        raise ValueError("A must be a 3x3 matrix")
    
    cofactors = np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            #Remove row i and column j
            minor = np.delete(np.delete(A,i,axis=0),j,axis=1)
            
            #Cofactor = sign x determinant of minor
            cofactors[i,j] = (-1)**(i+j)*np.linalg.det(minor)

    return cofactors.T 


A = np.array([
    [1,2,3],
    [0,1,4],
    [5,6,0]
],dtype=float)

adj = adjugate(A)
print(adj)

equality = np.allclose(adj/np.linalg.det(A),np.linalg.inv(A))