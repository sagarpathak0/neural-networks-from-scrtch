#Using OOPS for input outputs
import numpy as np

np.random.seed(0)
#Input Layer
X = [[1 , 2, 3, 2.5],
    [2.0 , 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]]

def create_data(points, classes):
    X = np.zeros((points*classes,2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 3)

#Hidden Layer
#Weights are generally random numbers between -1 and 1
#In general biases are zero
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#Activation Function(ReLU)
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()
layer1.forward(X)

activation1.forward(layer1.output)

print(activation1.output)
