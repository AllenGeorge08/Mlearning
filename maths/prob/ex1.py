# Implement inverse transform sampling for the exponential distribution. Verify by sampling 10,000 values and comparing the histogram to the true PDF.

# Inverse transform: 
import math
import random 
import matplotlib.pyplot as plt


# lambda=2

def pdf_exponential_distribution(lam,x):
    ans = lam*math.pow(math.e,-lam*x)
    return ans 


print("PDF :", pdf_exponential_distribution(2,0.602))

# u=0.7 -> inv cdf  -> res
# Inverse cdf
def exponential_distribution_inv_transform(lam):
    u = random.random()
    num = -math.log(1-u)
    return num/lam


# for 0.602

# res = exponential_distribution_inv_transform(2)
# print("Inverse CDF: ",res)


def inv_cdf_to_cdf(lam,x):
    return 1-math.exp(-lam*x)


# print("CDF derived from inverse cdf result: ",inv_cdf_to_cdf(2,res))
samples = [exponential_distribution_inv_transform(2) for _ in range(10000)]

plt.hist(samples,color='skyblue',edgecolor='black',bins=50,density=True)

import numpy as np
# True pdf 
x = np.linspace(0,4,500)
pdf = 2*np.exp(-2*x) #2 = lambda

plt.plot(x,pdf)
plt.title("Samples vs true exponential pdf")
plt.savefig('PDF.png')