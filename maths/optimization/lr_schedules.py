#Step-decay
lr = 0.1
gamma = 0.1

for epoch in range(30):
    if epoch>0 and epoch%10 == 0:
        lr *= gamma 
    
    print(epoch,lr)


# Exponential Decay
print("-"*50)
print("Exponential Decay")
gamma_exp = 0.95 

for epoch in range(10):
    current_lr = lr*(gamma**epoch)
    print(epoch,"-> ",current_lr)


