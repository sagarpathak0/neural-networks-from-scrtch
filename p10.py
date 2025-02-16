#Categorical Cross entropy loss
import math
import numpy as np

#Higher the confidence lower the loss
#means log(0.7) will be lower than log(0.2)
sf_out = [0.7,0.1,0.2]
sf_label = [1,0,0] # one-hot encoded label
target_class = 0

# Corrected loss calculation for one-hot encoded label
loss = -math.log(sf_out[target_class])
print("Loss for single sample:", loss)

sf_out = np.array([[0.7,0.1,0.2],
          [0.1,0.5,0.4],
          [0.02,0.9,0.08]])
class_targets = np.argmax(sf_out, axis=1, keepdims=True)

print(-np.log(sf_out[[0,1,2], class_targets]))