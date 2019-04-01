import pylab
import numpy as np
import matplotlib.pyplot as plt
import threading
import concurrent.futures
from funciones import *
from DCOV_IndependenceTest import *
power1 = np.zeros(5)
power2 = np.zeros(5)
ruidos = np.linspace(0,3,5)
for j,ruido in enumerate(ruidos):
  for i in range(100):
    x = np.random.rand(1000)
    z = Linear(x)
    y = np.array([j for j in z]) + np.random.normal(0,ruido,1000)
    x = (x - np.mean(x))*1./np.std(x)
    y = (y - np.mean(y))*1./np.std(y)
    [Sdcov,Sdcor,pvalue] = dcov(x,y,0.05,R = 100)
    if pvalue < 0.05:
      power1[j] +=0.01
    [Sdcov,Sdcor,statistic,thresh] = dcov(x,y,0.05)
    if statistic > thresh:
      power2[j] +=0.01
  print(ruido,end="\r")
plt.plot(ruidos,power1,'b')
plt.plot(ruidos,power2,'r')
plt.show()
np.savetxt("datos/DCOVPermsVsAsymtoticLinear.txt",np.stack((power1,power2)),delimiter="\t")