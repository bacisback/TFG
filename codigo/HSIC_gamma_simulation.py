import numpy as np
from HSIC_IndependenceTest import *
import scipy.stats as stats
import matplotlib.pyplot as plt
sim = 500
figure,ax1 = plt.subplots(1,1, sharey=True)
histogram = np.zeros(sim)
for i in range(sim):
  x,y = np.random.multivariate_normal([0,0], np.eye(2), 1000).T
  [histogram[i],_]=HSIC_test_gamma(x,y,0.05)
fit_alpha,fit_loc,fit_beta = stats.gamma.fit(histogram)
histogram = np.sort(histogram)
ax1.hist(histogram,density=True, histtype='step',label="HSIC")
ax1.plot(histogram,stats.gamma.pdf(histogram,fit_alpha,fit_loc,fit_beta),label="Gamma")
ax1.set_title("HSIC statistic under H0")
plt.legend(loc='best')
plt.show()