import numpy as np
inputs = [1.2, 5.1, 2.1]
weight = [3.1, 2.1, 8.7]
bias = 3
outputs = 0

for i in range(0,3):
    outputs+=inputs[i]*weight[i]

outputs+=bias
print(outputs)

op = np.dot(weight,inputs) + bias
print(op)