import numpy as np
import matplotlib.pyplot as plt
from funciones import *
functions = [Linear,Parabolic,Cubic,Sin1,Sin2,root4,circle,step,xsin,logarithm,gausian,twod_gaussian]
titles = ["lineal","Parabolic","Quadratic","Sin(4pix)","Sin(16pix)","fourth root","circle","step","xsin(x)","logarithm","gausian","2D gausian"]
n = 200
fig,ax = plt.subplots(4,3,sharey=True)
for i in range(11):
  x = np.random.rand(n)
  z = functions[i](x)
  y = np.array([j for j in z])
  x = (x - np.mean(x))*1./np.std(x)
  y = (y - np.mean(y))*1./np.std(y)
  ax = plt.subplot(4,3,i+1)
  plt.plot(x,y,'d')
  ax.set_title(titles[i])
x,y = bivariate_gaussian(200,cov = np.eye(2))

ax = plt.subplot(4,3,12)
plt.plot(x,y,'d')
ax.set_title(titles[11])
plt.show()