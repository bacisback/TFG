import numpy as np
import matplotlib.pyplot as plt
from funciones import *
titulos = ["bivariate gaussian","Gaussian multiply uniform", "Mixture of 3 gaussians","Gaussian multiplicative noise"]
funciones = [bivariate_gaussian,gaussian_multiply_uniform,mixture_3_gaussians,gaussian_multiplicative_noise]
fig,ax = plt.subplots(2,2,sharey=True)
for i in range(4):
  ax[int(i/2),np.mod(i,2)].set_title(titulos[i])
  x,y = funciones[i](200)
  x = (x - np.mean(x))*1./np.std(x)
  y = (y - np.mean(y))*1./np.std(y)
  ax[int(i/2),np.mod(i,2)].plot(x,y,'d')
plt.show()