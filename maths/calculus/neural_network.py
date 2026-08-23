import numpy as np 

def sigmoid(z):
    return 1/(1+np.exp(-z))


def train(X,Y,epochs=1000,lr=0.1):
    w = 0.0 
    b = 0.0

    for epoch in range(epochs):
        total_loss = 0 
        for x,y in zip(X,Y):
            #Forward propagation
            z = w*x+b 
            a = sigmoid(z)


           #Loss
            eps = 1e-8 
            #Binary cross entropy loss
            loss = -(y*np.log(a+eps)+(1-y)*np.log(1-a+eps))

            total_loss += loss 

            # Backward propagation
            dz = a-y
            dw = dz*x  
            db = dz 

            # Update 
            w -= lr*dw 
            b -= lr*db 

        
        if epoch %100 == 0:
            print(epoch,total_loss/len(X),w,b)

    
    return w,b



X = np.array([0,1,2,3])
Y = np.array([0,0,1,1])

w,b = train(X,Y)
print(w,b)


            

