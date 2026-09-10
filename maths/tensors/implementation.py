import numpy as np 

x = np.array([
    [1,2,3],
    [4,5,6]
])

print(x)
print(f'Shape : {x.shape}')
print(f'Rank: {x.ndim}')
print(f'Elements : {x.size}')


# Reshape
y = np.arange(1,13)
print(y)

y = y.reshape(3,4)
print(y.shape)

print("-"*50)
z = np.arange(1,13)
missing_dim = z.reshape(-1,4)
# print(missing_dim)
print(missing_dim)


# Squeezing 
print("-"*50)
squeeze = np.zeros((1,3,1,2))
print(f"Squeeze before squeezing : {squeeze}")
print(squeeze.shape)


squeeze = np.squeeze(squeeze)
print(f"Now squeezed x : {squeeze}")
print(squeeze.shape)

# Unsqueeze  (Inserting a dimenssion of size 1)
shape = np.array([1,2,3])

shape =np.expand_dims(shape,axis=0)
print(f"Unsqueezed: {shape.shape}")
print(f"shape: {shape.shape}")

# shape  =np.expand_dims(shape,axis=1)
