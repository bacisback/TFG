import numpy as np
import matplotlib.pyplot as plt
titles = ["bivariate gaussian","Gaussian multiply uniform", "Mixture of 3 gaussians","Gaussian multiplicative noise"]
figure,ax = plt.subplots(2,2, sharey=True)
solutions = np.loadtxt("./datos/testPerms/dcov500.txt",delimiter="\t")
solutions_As = np.loadtxt("./datos/testPerms/Asymptotic/DCOV.txt",delimiter="\t")
plt.suptitle("DCOV")
for i in range(2):
  for j in range(2):
    ax[i,j].set_ylim((-0.1,1.1))
    ax[i,j].set_title(titles[i*2+j])
    ax[i,j].plot(np.linspace(10,500,7).astype(int) ,solutions[i*2+j,:],label="Real")
    ax[i,j].plot(np.linspace(10,500,7).astype(int),solutions_As[i*2+j,:]+0.01,label="Asymptotic")
    ax[i,j].legend(loc='best')
figure,ax = plt.subplots(2,2, sharey=True)
solutions = np.loadtxt("./datos/testPerms/rdc500.txt",delimiter="\t")
solutions_As = np.loadtxt("./datos/testPerms/Asymptotic/RDC.txt",delimiter="\t")
plt.suptitle("RDC")
for i in range(2):
  for j in range(2):
    ax[i,j].set_ylim((-0.1,1.1))
    ax[i,j].set_title(titles[i*2+j])
    ax[i,j].plot(np.linspace(10,500,7).astype(int) ,solutions[i*2+j,:],label="Real")
    ax[i,j].plot(np.linspace(10,500,7).astype(int),solutions_As[i*2+j,:],label="Asymptotic")
    ax[i,j].legend(loc='best')
figure,ax = plt.subplots(2,2, sharey=True)
solutions = np.loadtxt("./datos/testPerms/hsic500.txt",delimiter="\t")
solutions_As = np.loadtxt("./datos/testPerms/Asymptotic/HSIC.txt",delimiter="\t")
plt.suptitle("HSIC")
for i in range(2):
  for j in range(2):
    ax[i,j].set_ylim((-0.1,1.1))
    ax[i,j].set_title(titles[i*2+j])
    ax[i,j].plot(np.linspace(10,500,7).astype(int) ,solutions[i*2+j,:],label="Real")
    ax[i,j].plot(np.linspace(10,500,7).astype(int),solutions_As[i*2+j,:],label="Asymptotic")
    ax[i,j].legend(loc='best')
plt.show()