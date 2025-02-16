import numpy as np
inputs = [1 , 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
bias = [2,3,0.5]
# outputs = []

# for weight_l,biass in zip(weights,bias):
#     # for ip,weight in zip(inputs,weight_l):
#     #     output += ip * weight
#     #outputs.append(output + biass)
#     #this is the nupy code for above code
#     output = np.dot(inputs,weight_l) + biass
#     outputs.append(output)

#for the above code we can write
outputs = np.dot(weights, inputs) + bias #Dot cannot be other way around
print(outputs)