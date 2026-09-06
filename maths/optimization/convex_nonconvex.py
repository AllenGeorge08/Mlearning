# L(w)=w**4−w**2

def loss(w):
    return w**4-w**2

def gradient(w):
    return 4*w**3 - 2*w


w = 0.5
lr = 0.05 

for step in range(20):
    grad = gradient(w)
    w = w-lr*grad 

    print(f"Step = {step} ->  w={w:.4f} -> loss={loss(w):.4f}")

