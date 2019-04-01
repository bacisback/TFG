import numpy as np
import matplotlib.pyplot as plt
titles = ["bivariate gaussian","Gaussian multiply uniform", "Mixture of 3 gaussians","Gaussian multiplicative noise"]
figure,ax = plt.subplots(2,2, sharey=True)
solutions = np.loadtxt("./datos/Varing_Size10-500DCOV.txt",delimiter="\t")
for i in range(2):
  for j in range(2):
    ax[i,j].set_ylim((-0.1,1.1))
    ax[i,j].set_title(titles[i*2+j])
    ax[i,j].plot(np.linspace(10,500,5).astype(int) ,solutions[i*2+j,:],label="DCOV")
solutions = np.loadtxt("./datos/Varing_Size10-500HSIC.txt",delimiter="\t")
for i in range(2):
  for j in range(2):
    ax[i,j].set_ylim((-0.1,1.1))
    ax[i,j].set_title(titles[i*2+j])
    ax[i,j].plot(np.linspace(10,500,5).astype(int) ,solutions[i*2+j,:],label="HSIC")
solutions = np.loadtxt("./datos/Varing_Size10-500RDC.txt",delimiter="\t")
for i in range(2):
  for j in range(2):
    ax[i,j].set_ylim((-0.1,1.1))
    ax[i,j].set_title(titles[i*2+j])
    ax[i,j].plot(np.linspace(10,500,5).astype(int) ,solutions[i*2+j,:],label="RDC")
    ax[i,j].legend(loc='best')
plt.show()