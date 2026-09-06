class Adam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.epsilon = epsilon 


        self.m= 0 
        self.v = 0 
        self.t  = 0
    

    def step(self,w,gradient):
        self.t += 1
        self.m = (self.beta1*self.m + (1-self.beta1)*gradient)  #momentum
        
        #second moment 
        self.v = (self.beta2*self.v) + (1-self.beta2)*gradient**2  #adagrad

        # generally B1 = 0.9 B2 = 0.99

        #Biass correction
        m_hat = self.m/(1-self.beta1**self.t)
        v_hat = self.v/(1-self.beta2**self.t)
        
        w  = w-self.lr*m_hat/(v_hat**0.5 + self.epsilon)

        return w 