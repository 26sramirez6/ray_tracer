'''
Created on Jul 17, 2018

@author: 26sra
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

N = int(sys.argv[1])

with open("./hw4.out", "rb") as f:
    data = np.fromfile(f, dtype=np.double, count=-1)
    data = data.reshape(N, N)
    print(data)
    print(data.sum())
    plt.imshow(data)
    plt.savefig("hw4.svg")
