import math
import torch
import random

# Compute the cross-entropy loss for a 5-class classifier that outputs logits [2.0, 0.5, -1.0, 3.0, 0.1] when the correct class is index 3. Then verify your answer with PyTorch's nn.CrossEntropyLoss.


logits = [2.0, 0.5, -1.0, 3.0, 0.1]

def softmax(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)

    prob = [exp/total for exp in exps]
    return prob  


def cross_entropy(logits,target_index):
    probs = softmax(logits)
    target = probs[target_index]
    answer = -(math.log(target))
    return answer 


import torch.nn as nn 

final_loss = cross_entropy(logits,3)
print(f"Final loss naive version : {final_loss}")


print("PyTorch Version")
logits_tensor = torch.tensor(logits)
target = torch.tensor([3])
loss_fn = nn.CrossEntropyLoss()
print(loss_fn(logits_tensor,target))


