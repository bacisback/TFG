import numpy as np
from DCOV_IndependenceTest import *
from scipy.stats import norm
import matplotlib.pyplot as plt
histogram = np.loadtxt("./datos/DCOVhistograma.txt",delimiter="\t")
figure,[ax1,ax2] = plt.subplots(1,2, sharey=True)
x = np.linspace(0,1,100)
ax1.hist(histogram[0],density=True, histtype='step',label="DCOV")
ax1.plot(x,norm.pdf(x)**2,label="norm")
ax1.set_title("DCOV statistic under H0")
ax2.hist(histogram[1],density=True, histtype='step',label="DCOV")
ax2.plot(x,norm.pdf(x)**2,label="norm")
ax2.set_title("DCOV statistic under H1")
plt.legend(loc='best')
plt.show()