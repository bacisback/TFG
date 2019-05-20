import numpy as np
import matplotlib.pyplot as plt
from funciones import *
ninit = 0
nend = 1
n = 8
angle=np.linspace(ninit,nend,n)
size = 200
fig,ax = plt.subplots(2,4,sharey=True)
for i in range(8):
  x,y = rotation(size,angle[i]*np.pi)
  x = (x - np.mean(x))*1./np.std(x)
  y = (y - np.mean(y))*1./np.std(y)
  ax[int(i/4),np.mod(i,4)].plot(x,y,'d')
  ax[int(i/4),np.mod(i,4)].set_title(str(angle[i])+"pi")
plt.show()
