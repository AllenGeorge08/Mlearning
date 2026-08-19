import math
import numpy  as np

def magnitude(v):
    total = 0

    for x in v:
        total += x*x

    return math.sqrt(total)

v = [3,4]
print(magnitude(v))

v = np.array([3,4])
print(np.linalg.norm(v))



def add_vectors(a,b):
    result = []
    for i in range(len(a)):
        result.append(a[i]+b[i])
    return result 


print(add_vectors([2,3],[5,7]))

a = np.array([2,1])
b  = np.array([1,3])
print(a+b)

def scalar_multiply(c,v):
    return [c*x for x in v]

print(scalar_multiply(3,[3,3,4]))

v2 = np.array([2,1])
print(3*v2)


def dot(a,b):
    total = 0
    for i in range(len(a)):
        total += a[i]*b[i]
    return total 


print(dot([2,1],[3,3]))


print(np.dot([2,1],[5,6]))


# Cosine similarity
def cosine_similarity(a,b):
    dot_product = np.dot(a,b)

    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))

    return dot_product/(mag_a*mag_b)


print(cosine_similarity([3,4],[6,8]))

# Matrix x vector
A = np.array([[1,2],[3,4]])
x= np.array([5,6])

print(A@x)


# Matrix multiplication
K = np.array([[1,2],[78,90]])
M = np.array([[56,78],[90,90]])

print(K@M)

# Transpose
def transpose(A):
    print([list(row) for row in zip(*A)])
    return [list(row) for row in zip(*A)]

print(transpose(K))

# Numpy transpose
print(K.T)


# Normalization v (unit) = v/||v||
def normalize(v):
    magnitude = sum(x*x for x in v)**0.5 
    return [x/magnitude for x in v]

print(normalize([3,4]))


# Numpy
v5 = np.array([3,4])
normalized = v/np.linalg.norm(v)
print(normalized)

#Linear Independence

def are_linearly_independent(vectors):
    A = np.column_stack(vectors) #convert to matrix
    return np.linalg.matrix_rank(A) == len(vectors)


v3 = np.array([1,0])
v9 = np.array([0,1])

print(are_linearly_independent([v3,v9]))


# Low rank
A = np.random.randn(4096,8)
B = np.random.randn(8,4096)

# Matrix mul
delta_W = A@B 
print(delta_W.shape)
# print(np.linalg.matrix_rank(delta_W)) #Lora rank


sqrt2 = np.sqrt(2)
q1 = np.array([1,1])/sqrt2 
q2 = np.array([-1,1])/sqrt2

Q = np.column_stack([q1,q2])
print("Q:" ,Q)