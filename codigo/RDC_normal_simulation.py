import numpy as np
from RDC_IndependenceTest import *
from scipy.stats import chi2
import matplotlib.pyplot as plt
histogram = np.loadtxt("./datos/RDChistograma.txt",delimiter="\t")
histogram[0] = np.sort(histogram[0])
histogram[1] = np.sort(histogram[1])
figure,[ax1,ax2,ax3] = plt.subplots(1,3, sharey=True)
x = chi2.fit(histogram[0])
ax1.hist(histogram[0],density=True, histtype='step',label="HSIC")
ax1.plot(histogram[0],chi2.pdf(histogram[0],x[0],x[1],x[2]))
ax1.set_title("RDC statistic under H0")
x = chi2.fit(histogram[1])
ax2.hist(histogram[1],density=True, histtype='step',label="HSIC")
ax2.plot(histogram[1],chi2.pdf(histogram[1],x[0],x[1],x[2]))
ax2.set_title("RDC statistic under H0")
ax3.hist(histogram[0],density=True, histtype='step',label="HSIC")
plt.legend(loc='best')

plt.show()