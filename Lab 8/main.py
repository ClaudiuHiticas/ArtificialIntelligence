import numpy as np
import matplotlib.pyplot as mpl
from copy import deepcopy 

def linear(x):
    return x

def dLinear(x):
    return 1


def readFromFile(fileName):
    f = open(fileName)
    data = []
    for row in f:
        row = row.split(' ')
        vec = [1]
        for idx in range(len(row)):
            vec.append(float(row[idx]))
        data.append(vec)
    f.close()
    return data
    

def mix(data):
    np.random.shuffle(data)
    result = []
    for idx in range(len(data)):
        result.append([data[idx].pop()])
    data = np.array(data)
    result = np.array(result)
    return data, result


class NeuralNetwork:    
    def __init__(self, x, y, hidden):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], hidden) 
        self.weights2   = np.random.rand(hidden, 1)            
        self.y          = y
        self.output     = []
        self.loss       = []
        self.m          = x.shape[0]

    def feedforward(self):
        self.layer1 = linear(np.dot(self.input, self.weights1))
        self.output = linear(np.dot(self.layer1, self.weights2))
        
    def backprop(self,l_rate):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * dLinear(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * dLinear(self.output), self.weights2.T) * dLinear(self.layer1)))
        
        self.weights1 += (1/self.m) * l_rate * d_weights1
        self.weights2 += (1/self.m) * l_rate * d_weights2
        self.loss.append(sum((self.y - self.output)**2))


data = readFromFile('data.csv')
train = data[:200]
test = data[200:]
testing = deepcopy(train)
X, Y = mix(testing)
print(X.shape)
print(Y.shape)
x, y = mix(test)
print(x.shape)
nn = NeuralNetwork(X,Y,2)

nn.loss=[]
iterations =[]
for i in range(4000):
    nn.feedforward()
    nn.backprop(0.0001)
    iterations.append(i)
    another = deepcopy(train)
    X,Y = mix(another)
    nn.input = X
    nn.y = Y

print(nn.output)
mpl.plot(iterations, nn.loss, label='loss value vs iteration')
mpl.xlabel('Iterations')
mpl.ylabel('loss function')
mpl.legend()
mpl.show()
