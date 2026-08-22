import numpy as np

def eigenvalues_2x2(matrix):
    a,b = matrix[0]
    c,d = matrix[1]

    print(f"Matrix[0] decoupled is : {a},{b}")
    print(f"Matrix[1] decoupled is : {c},{d}")
    trace = a+d 
    det = a*d - b*c 

    discriminant = trace**2 - 4*det 
    if discriminant <0:
        real = trace/2 
        imag = (-discriminant)**0.5 /2 
        return (complex(real,imag), complex(real,-imag))

    
    root = discriminant**0.5
    return ((trace+root)/2, (trace-root)/2)


A = np.array([
    [1,2],
    [3,4]
])

# print(eigenvalues_2x2(A))


# def eigenvector_2x2(matrix,eigenvalue):
#     a,b = matrix[0]
#     c,d = matrix[1]

#     if abs(b) > 1e-10:
#         v = [b,eigenvalue-a]
#     elif abs(c)>1e-10:
#         v = [eigenvalue-d,c]
#     else:
#         if abs(a-eigenvalue)<1e-10:
#             v = [1,0]
#         else:
#             v = [0,1]
    
#     mag = (v[0]**2 + v[1]**2)**0.5 

#     return [v[0]/mag,v[1]/mag]


# ev1,ev2 = eigenvalues_2x2(A)
# (evec1,evec2) = eigenvector_2x2(ev1,A)
# print(evec1,evec2)


values,vectors = np.linalg.eig(A)
print(f"Eigen Values: {values} , Eigen Vectors : {vectors[:,0]}")


def rotation_3d_z(theta):
    c,s = np.cos(theta),np.sin(theta)
    return np.array([
        [c,-s,0],
        [s,c,0],
        [0,0,1]
    ])

