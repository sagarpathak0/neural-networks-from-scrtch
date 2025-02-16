#Softmax Activation Function-Exponential+Normalize
#Exponential function
import math
import numpy as np
l_ip = [4.8, 1.21, 2.385]
E = math.e

# exp_Values = []

# for i in l_ip:
#     exp_Values.append(E**i)
#For the above code we can do
exp_Values = np.exp(l_ip)

print(exp_Values)

#Normalization - probability Distribution
# norm_base = sum(exp_Values)
# norm_values = []
# for i in exp_Values:
#     norm_values.append(i/norm_base)
#For the above code we can do
norm_values = exp_Values/ np.sum(exp_Values)

print(norm_values)
