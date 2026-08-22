import numpy as np

angles = np.linspace(0,2*np.pi,8,endpoint=False)
points = np.column_stack((np.cos(angles),np.sin(angles)))

print("Original Points...")
print(points)

def rotate_2d(theta):
    rad = np.radians(theta)
    return np.array([
        [np.cos(rad),-np.sin(rad)],
        [np.sin(rad),np.cos(rad)]
    ])


def scaling_2d(sx,sy):
    return np.array([
        [1,sx],
        [sy,1]
    ])


def shearing_2d(sx,sy=0.0):
    return np.array([
        [1,sx],
        [sy,1]
    ])


R = rotate_2d(30)
S = scaling_2d(1.5,0.8)
N = shearing_2d(0.3,0.0)

M = N@S@R  #trnasformations from right to left
# Apply matrix ops to pointss

transformed_pts = []
for point in points:
    new_point = M@point 
    transformed_pts.append(new_point)

transformed_points = np.array(transformed_pts)
print("Transformed points...")
print(transformed_points)

det_R = np.linalg.det(R)
det_S = np.linalg.det(S)
det_N = np.linalg.det(N)

det_product = det_R*det_S*det_N
det_composed = np.linalg.det(M)

print("Is equal =" ,np.isclose(det_product,det_composed))