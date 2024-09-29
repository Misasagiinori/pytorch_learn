import math
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l

#自己用matplotlib重新实现李沐老师p91页d2l实现正态分布绘图。


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

x = np.arange(-7, 7, 0.001)
#mu=0
#sigma=1
params = [(0, 1), (0, 2), (3, 1)]
for mu,sigma in params:
    y = normal(x, mu, sigma) 
    plt.plot(x, y,label=f'mean {mu}, std {sigma}')
    plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()  
