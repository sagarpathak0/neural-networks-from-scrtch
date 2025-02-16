import numpy as np

np.random.seed(0)

# Create Dataset
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 3)

# Fully Connected Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# Activation Function: ReLU
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0  # Zero out gradients for non-positive inputs

# Activation Function: Softmax
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Loss Function: Categorical Cross-Entropy
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

# Softmax Activation + Cross-Entropy Loss Combined
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def forward(self, inputs, y_true):
        self.activation = Activation_Softmax()
        self.activation.forward(inputs)
        self.output = self.activation.output
        return Loss_CategoricalCrossentropy().calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 1:
            y_true = np.eye(dvalues.shape[1])[y_true]

        self.dinputs = (dvalues - y_true) / samples

#OPTIMIZERS

# Simple SGD Optimizer
class Optimizer_SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

# Momentum SGD Optimizer
class MOptimizer_SGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_momentums = {}  # To store past weight updates
        self.bias_momentums = {}  # To store past bias updates

    def update(self, layer):
        if layer not in self.weight_momentums:
            self.weight_momentums[layer] = np.zeros_like(layer.weights)
            self.bias_momentums[layer] = np.zeros_like(layer.biases)

        # Compute momentum updates
        self.weight_momentums[layer] = (
            self.momentum * self.weight_momentums[layer] - self.learning_rate * layer.dweights
        )
        self.bias_momentums[layer] = (
            self.momentum * self.bias_momentums[layer] - self.learning_rate * layer.dbiases
        )

        # Apply updates
        layer.weights += self.weight_momentums[layer]
        layer.biases += self.bias_momentums[layer]



# Optimizer: Adam
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        self.m = {}
        self.v = {}

    def update(self, layer):
        layer_id = id(layer)  # Unique identifier for each layer

        # Initialize moment estimates for this layer if not present
        if layer_id not in self.m:
            self.m[layer_id] = np.zeros_like(layer.weights)
            self.v[layer_id] = np.zeros_like(layer.weights)

        self.t += 1  # Increment time step

        # Apply learning rate decay
        self.current_learning_rate = self.learning_rate / (1 + self.decay * self.t)

        # Compute moving averages of gradients
        self.m[layer_id] = self.beta1 * self.m[layer_id] + (1 - self.beta1) * layer.dweights
        self.v[layer_id] = self.beta2 * self.v[layer_id] + (1 - self.beta2) * (layer.dweights**2)

        # Bias correction
        m_hat = self.m[layer_id] / (1 - self.beta1**self.t)
        v_hat = self.v[layer_id] / (1 - self.beta2**self.t)

        # Update weights and biases
        layer.weights -= self.current_learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        layer.biases -= self.current_learning_rate * layer.dbiases

# Initialize layers
dense1 = Layer_Dense(2, 128)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.030)

# Training loop
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    
    loss = loss_activation.forward(dense2.output, y)

    # Accuracy calculation
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dense1.backward(dense2.dinputs)

    # Update weights and biases
    optimizer.update(dense2)
    optimizer.update(dense1)

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
