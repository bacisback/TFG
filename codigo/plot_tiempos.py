import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def func1(x,a,b,c):
  return a*x**2 + b*x + c
def func2(x,c,k):
  return k**2*x + c
dcov = np.loadtxt("./datos/TIMES/DCOVtiempos.txt")
rdc = np.loadtxt("./datos/TIMES/RDCtiempos.txt")
hsic = np.loadtxt("./datos/TIMES/HSICtiempos.txt")
division = np.linspace(10,1000,20)

# plt.plot(division,dcov[1,:]/25,'cyan', label ="DCOV's Mean")
# plt.plot(division,dcov[0,:]/25,'b', label ="DCOV's First Quantile")
# plt.plot(division,dcov[2,:]/25,'b', label ="DCOV's Third Quantile")

plt.plot(division,rdc[1,:],'lime', label ="RDC's Mean")
#plt.plot(division,rdc[0,:],'g', label ="RDC's First Quantile")
#plt.plot(division,rdc[2,:],'g', label ="RDC's Third Quantile")

plt.plot(division,hsic[1,:],'lightcoral', label ="HSIC's Mean")
#plt.plot(division,hsic[0,:],'r', label ="HSIC's First Quantile")
#plt.plot(division,hsic[2,:],'r', label ="HSIC's Third Quantile")
#print((1000,hsic[1,-1]))
#plt.annotate(str(hsic[1,-1]),xy=(1000,hsic[1,-1]),xytext = (900,0.145),arrowprops=dict(facecolor='black', shrink=5),)
#plt.annotate(str(rdc[1,-1]),xy=(1000,rdc[1,-1]),xytext = (900,0.050),arrowprops=dict(facecolor='black', shrink=5),)

popt, pcov = curve_fit(func1, division, hsic[1,:])
plt.plot(division,func1(division,*popt),'dr-',label="HSIC aproximation by a quadratic")

popt, pcov = curve_fit(func2, division[:10], rdc[1,0:10])
plt.plot(division[:10],func2(division[:10],*popt),'dg-',label="RDC aproximation by a line with slope of 9")
popt, pcov = curve_fit(func2, division[10:], rdc[1,10:])
plt.plot(division[10:],func2(division[10:],*popt),'dg-',label="RDC aproximation by a line with slope of 16")
plt.legend(loc="best")

plt.show()