from math import isclose
import numpy as np 

def gram_schmidt(vectors):
    orthogonal = []

    for v in vectors:
        u = v.astype(float).copy()

        #Remove proj onto prev vectors
        for previous in orthogonal:
            projection = (
                np.dot(v,previous)/np.dot(previous,previous)
            )*previous

            u -= projection 
        
        #Normalize
        norm = np.linalg.norm(u)

        if np.isclose(norm,0):
            raise ValueError("Vectors are linearly dependent")
        
        u = u/norm 
        orthogonal.append(u)

    return np.array(orthogonal)


v1 = np.array([1.0,1.0])
v2 = np.array([1.0,0.0])

Q = gram_schmidt([v1,v2])
print(Q)

print(Q@Q.T)


# QR IMplementation
A = np.array([
    [1.0,1.0],
    [1.0,0.0]
])

Q,R = np.linalg.qr(A)
print("Q: ")
print("\n R:")
print(R)

print(np.allclose(A,Q@R))
