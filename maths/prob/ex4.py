import math

sentence_probs = [0.01]*50
log_probs = [math.log(prob) for prob in sentence_probs]
print(log_probs)

def function(log_probs):
    # log 10  = 2, 10=e**2
    total_log_prob = sum(log_probs)
    raw_prob = math.exp(total_log_prob)
    return total_log_prob,raw_prob 


print(function(log_probs)) 




