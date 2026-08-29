import random
import math

samples = int(math.pow(10,5))
print(samples)

samples_A = [random.betavariate(51,951) for _ in range(samples)]
samples_B = [random.betavariate(66,936) for _ in range(samples)]

wins = 0 

for a,b in zip(samples_A,samples_B):
    if b>a:
        wins += 1

prob_B_better = wins/samples
print(prob_B_better)