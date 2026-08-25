import math 

class Value:
    def __init__(self,data,children=(),op=''):
        self.data = data 
        self.grad = 0.0 
        self._backward = lambda: None 
        self._prev = set(children)
        self._op = op 

    
    def __pow__(self,n):
        out = Value(self.data**n,(self,),f'**{n}')
        def _backward():
            self.grad += out.grad * n*(self.data**(n-1))
        out._backward = _backward
        return out  
        
    
    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f},grad={self.grad:.4f})"

    # (2,1)(3,0) , original x and then the derivative
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data,(self,other),'+') 
        def _backward():  
            self.grad += out.grad 
            other.grad += out.grad 
        out._backward = _backward
        return out 

    
    def __radd__(self,other):
        return self.__add__(other)

    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad += out.grad * other.data 
            other.grad += self.data * out.grad 
        
        out._backward=_backward
        return out 

    
    def __rmul__(self,other):
        return self.__mul__(other)

    
    def __neg__(self):
        return self*-1

    def __sub__(self,other):
        return self+(-other)

    def __rsub__(self,other):
        return other+(-self)

    
    def __pow__(self,n):
        out = Value(self.data**n,(self,),f'**{n}')
        def _backward():
            self.grad += n*(self.data**(n-1))*out.grad 
        out._backward = _backward
        return out 

    
    def __truediv__(self,other):
        return self*(other**-1) if isinstance(other,Value) else self*(Value(other)**-1)
    

    def relu(self):
        out = Value(max(0,self.data),(self,),'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0)*out.grad 
        out._backward = _backward
        return out 

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1-t**2)*out.grad 
        out._backward = _backward
        return out 

    def exp(self):
        e = math.exp(self.data)
        out = Value(e,(self,),'exp')
        def _backward():
            self.grad += e*out.grad 
        out._backward = _backward
        return out 

    #math.log(original data) and then the self.gradient = 
    def log(self):
        out = Value(math.log(self.data),(self,),'log')
        def _backward():
            self.grad += (1.0/self.data)*out.grad 
        out._backward = _backward
        return out 

    def backward(self):
        topo =[]
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad += 1.0 
        for v in reversed(topo):
            v._backward()


#Exercise 1.

x = Value(2)
y = x.__pow__(3)
y.backward() # now y.grad / or the out.grad has to be incremented by 1
print(f"Current x data: {x.data}")
print(f"Y data after power of 3: {y.data}")
print(f"X grad after power of 3: {x.grad}")


#Exercise 2
a = Value(0)
b = Value(2)
ta = a.tanh()
ta.backward() #this is the other.grad that's being multiplied and we gotta increment it'ss value


tb=b.tanh()
tb.backward()
print(f"A grad : {a.grad}")
print(f"B grad : {b.grad}")


# a = x1 * x2          # a = 6.0
# b = a + Value(1.0)    # b = 7.0
# y = b.relu()          # y = 7.0

# y.backward()

# print(f"y = {y.data}")          # 7.0
# print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
# print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)

#Exercise 
# 3:

x1 = Value(-2.0)
x2 = Value(3.0)

w1 = Value(3.0)
w2 = Value(4.0)
b = Value(1.0)

y = (w1*x1 + w2*x2 + b).relu()
print(f"Y data before backward call : {y.grad }")
y.backward()
print(f"Y data after backward call : {y.grad }")

print(f"w1 grad = W1 {w1.grad}")
print(f"w2 grad = W1 {w2.grad}")
print(f"x1 grad = W1 {x1.grad}")
print(f"x2 grad = W1 {x2.grad}")
print(f"b grad = W1 {b.grad}")


import torch 

x1_t = torch.tensor(-2.0,requires_grad=True)
x2_t = torch.tensor(3.0,requires_grad=True)
w1_t = torch.tensor(3.0,requires_grad=True)
w2_t = torch.tensor(4.0,requires_grad=True)
b_t = torch.tensor(1.0,requires_grad=True)

y_t  = torch.relu(w1_t*x1_t + w2_t*x2_t + b_t)
y_t.backward()


print("-------Tensor Verification------")
print(f"w1 grad = W1 {w1_t.grad}")
print(f"w2 grad = W1 {w2_t.grad}")
print(f"x1 grad = W1 {x1_t.grad}")
print(f"x2 grad = W1 {x2_t.grad}")
print(f"b grad = W1 {b_t.grad}")



import random 

class Neuron:
    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
     
    
    
    def __call__(self, x):
        activation = self.b 

        # Adding bias to w*x
        for wi,xi in zip(self.w,x):
            activation += wi*xi 
        

        return activation.tanh()


#Layer of neurons that receive the same input vector, if layer=Layer(2,4) -> 2 inputs 4 neurons
class Layer:

    def __init__(self,nin,nout):
        self.neurons=[Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        return [n(x) for n in self.neurons]

    

class MLP:
    def __init__(self,nin,nouts):
        sizes = [nin] + nouts 
        self.layers = [Layer(sizes[i],sizes[i+1]) for i in range(len(nouts))]

    
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        
        return x 


# Input 2-> Layer 1(4) -> Layer2(1) -> Output
model = MLP(2,[4,1])
        

# The entire training process

# Input -> ForwardPass(x->MLP-> Y^) -> L=(y^-y)**2 -> BackPropagation (L->weights calculating dL/dwi) -> Gradient descent(w <- wi - learning_rate*dL/dwi)
# Then repeat


# layer = Layer(2,4)
# print(layer)

x1 = Value(2.0)
x2 = Value(3.0)

print(x1.grad)
print(x2.grad)
print("-"*10)

y = (x1*x2 + 1).relu()

y.backward()

print(x1.grad)
print(x2.grad)


# Forward mode...
class Dual:
    def __init__(self,val,eps=0.0):
        self.val = val
        self.eps = eps

    def __add__(self,other):
        other = other if isinstance(other,Dual) else Dual(other)
        return Dual(self.val+other.val,self.eps+other.eps)

    def __mul__(self,other):
        other = other if isinstance(other,Dual) else Dual(other)
        return Dual(self.val*other.val,self.val*other.eps + self.eps*other.val)#a'b +ab'

    
    def relu(self):
        if self.val > 0:
            return Dual(self.val,self.eps)
        else:
            return Dual(0.0,0.0)



def neuron(w1, x1, w2, x2, b):
    return (w1*x1 + w2*x2 + b).relu()

vals = dict(w1=0.5,x1=1.0,w2=-0.3,x2=2.0,b=0.1)

grads = {}

for var in vals:
    duals = {k: Dual(v,1.0 if k==var else 0.0) for k,v in vals.items()}
    y = neuron(**duals)
    grads[var]=y.eps

print(grads)