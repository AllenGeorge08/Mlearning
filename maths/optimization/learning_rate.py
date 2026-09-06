def loss(w):
    return w**2 

def gradient(w):
    return 2*w 

def train(lr,steps=20):
    w = 3.0

    for step in range(steps):
        w = w-lr*gradient(w)

    return w,loss(w)


for lr in [0.01,0.1,0.5,1.1]:
    w,final_loss = train(lr)
    print(f"lr={lr}",f"w={w:.4f}",f"loss={final_loss:.4f}")

    