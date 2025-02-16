#Using OOPS for input outputs
import numpy as np

np.random.seed(0)
#Input Layer
X = [[1 , 2, 3, 2.5],
    [2.0 , 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]]

#Hidden Layer
#Weights are generally random numbers between -1 and 1
#In general biases are zero
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)
