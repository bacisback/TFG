import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
real = np.loadtxt("./datos/rotations/experiment100samples.txt")
asymptDCOV = np.loadtxt("./datos/rotations/Asymptotic/pequeDCOV.txt")
asymptRDC = np.loadtxt("./datos/rotations/Asymptotic/pequeRDC.txt")
asymptHSIC = np.loadtxt("./datos/rotations/Asymptotic/pequeHSIC.txt")
xR = np.linspace(0,2,100)
xA = np.linspace(0,0.25,20)
figure,ax = plt.subplots(1,3, sharey=True)
print([xR<=0.25])
aux = real[-3,:]
ax[0].plot(xR[xR<=0.25],aux[xR<=0.25],'r',label='real')
ax[0].plot(xA[xA<=0.25],asymptHSIC[xA<=0.25],'b',label='asymptotic')
ax[0].set_title("HSIC")
ax[0].set_ylabel("Power")
ax[0].set_xlabel("Rotation angle")
aux = real[-2,:]
ax[1].plot(xR[xR<=0.25],aux[xR<=0.25],'r',label='real')
ax[1].plot(xA[xA<=0.25],asymptRDC[xA<=0.25],'b',label='asymptotic')
ax[1].set_title("RDC")
ax[1].set_xlabel("Rotation angle")
aux = real[-1,:]
ax[2].plot(xR[xR<=0.25],aux[xR<=0.25],'r',label='real')
ax[2].plot(xA[xA<=0.25],asymptDCOV[xA<=0.25],'b',label='asymptotic')
ax[2].set_title("DCOV")
ax[2].set_xlabel("Rotation angle")
plt.legend(loc="best")
plt.show()