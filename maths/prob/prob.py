import math 
import random 

random.seed(42)

def factorial(n):
    result = 1
    for  i in range(2,n+1):
        result*=i
    return result  

def combinations(n,k):
    return factorial(n)//(factorial(k)*factorial(n-k))


# print(combinations(10,3))
def conditional_probability(p_a_and_b,p_b):
    return p_a_and_b/p_b 

# p**k(1=p)**k
def bernoulli_pmf(k,p):
    return p if k==1 else (1-p)


def categorical_pmf(k,probs):
    return probs[k]

def poisson_pmf(k,lam):
    return (lam**k)*math.exp(-lam)/factorial(k)


def uniform_pdf(x,a,b):
    if a<=x<=b:
        return 1.0/(b-a)
    return 0.0 

def  normal_pdf(x,mu,sigma):
    coeff = 1.0/(sigma*math.sqrt(2*math.pi))
    exponent = -0.5*((x-mu)/sigma)**2 
    return coeff*math.exp(exponent)

def expected_value(values,probablities):
    return sum(v*p for v,p in zip(values,probablities))

def variance(values,prob):
    mu = expected_value(values,prob)
    return sum(p*(v-mu)**2 for v,p in zip(values,prob))

def sample_bernoulli(p,n=1):
    return [1 if random.random() <p else 0 for _ in range(n)]


def sample_categorical(probs,n=1):
    cumulative=[]
    total = 0
    for p in probs:
        total += p 
        cumulative.append(total)
    print(cumulative)
    samples =[]
    for _ in range(n):
        r = random.random()
        for i,c in enumerate(cumulative):
            if r<=c:
                samples.append(i)
                break 
    return samples 


print(sample_categorical([0.2,0.5,0.1]))
# print(sample_bernoulli(1))

