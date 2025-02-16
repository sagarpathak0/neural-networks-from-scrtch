#Batches for softmax
import math
import numpy as np
l_ip =[[4.8, 1.21, 2.385],
       [8.9,-1.81,0.2],
       [1.41,1.051,0.026]]
exp_Values = np.exp(l_ip)

# print(exp_Values)
# norm_values = exp_Values/ np.sum(exp_Values)
norm_values = exp_Values/np.sum(exp_Values, axis=1, keepdims=True)

print(norm_values)