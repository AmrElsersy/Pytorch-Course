import numpy as np
import torchvision
import torch
import torch.nn as nn
import pandas
import PIL

# compares array of previous layer (without applying softmax (without probabilites) ) 
# and the labels as indexes of the only 1 in the array
loss = nn.CrossEntropyLoss()


# each number represents the index of the 1 (and others = 0) 
# one-hoted encoded ... 1 ====  [0,1,0] (the index 1 value = 1) 
x = torch.tensor([1,2,0])
# prop = torch.softmax(x, dim =0)
y = torch.tensor([ [0.6,0.9,0.5], [2,3,5], [22,3, 2] ], dtype=torch.float32)

l1 = loss( y, x)

softmax_indexes_from_output = torch.max(y, 1)

print(l1.item())
print(softmax_indexes_from_output)

# https://www.youtube.com/watch?v=7q7E91pHoW4&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=11
# https://www.youtube.com/watch?v=3t9lZM7SS7k&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=12