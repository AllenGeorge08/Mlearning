import numpy as np 

unit_square = np.array(
    [[1,2]
    , [4,5]
    ])
# We had to return matrix beta
def rotation(theta):
    cos,sin = np.cos(theta),np.sin(theta)
    return [[cos,-sin],[sin,cos]]


# We're returning a scaling matrix wtf haha
def scaling_2d(sx,sy):
    return [
        [sx,0],
        [0,sy]
    ]

def shearing_2d(kx,ky):
    return [[1,kx],[ky,1]]


arr = np.array([
    [1,2],
    [4,5]
])

print(rotation(arr))



