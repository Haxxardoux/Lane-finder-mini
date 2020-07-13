import numpy as np
import torch


thing = np.arange(0, 100)
thing2 = np.arange(0, 100)

doof1 = thing.reshape(50, 2)
doof2 = thing2.reshape(50, 2)

print(doof1)

doof1 = thing[:, 0::10]
doof2 = thing2[:, 1::10]



print(doof1)
