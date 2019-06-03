import numpy as np
import matplotlib.pyplot as plt

titles = ["lineal","Parabolic","Cubic","Sin(4pix)","Sin(16pix)","fourth root","circle","step","xsin(x)","logarithm","gausian","2D gausian"]
figure,ax = plt.subplots(4,3, sharey=True)
figure.suptitle("DCOV")
solutions = np.loadtxt("./datos/exp1Asint/DCOV.txt",delimiter="\t")
solutions_as = np.loadtxt("./datos/permutaciones200_2dcov.txt",delimiter="\t")
print(solutions_as)
indep = np.loadtxt("./datos/permutacionesVSasint/DCOV-0.0-200version.txt",delimiter="\t")
x = np.array(range(len(solutions[0,:])))
x2 = np.array(range(len(solutions_as[0,:])))
for i in range(len(titles)):
	print(int(i/4),len(titles),i)
	ax[int(i/3),int(np.mod(i,3))].set_ylim((-0.1,1.1))
	ax[int(i/3),int(np.mod(i,3))].set_title(titles[i])
	ax[int(i/3),int(np.mod(i,3))].plot(x*3./10,solutions[i,:],label="Asymptotic")
	if (i != 11):
		ax[int(i/3),int(np.mod(i,3))].plot(x2*3./10,solutions_as[i,:]*100,label="Real")
	else:
		ax[int(i/3),int(np.mod(i,3))].plot(np.linspace(0,3,5),indep[1,:],label="Real")
	ax[int(i/3),int(np.mod(i,3))].legend(loc='best')

figure,ax = plt.subplots(4,3, sharey=True)
figure.suptitle("HSIC")
solutions = np.loadtxt("./datos/exp1Asint/HSIC.txt",delimiter="\t")
solutions_as = np.loadtxt("./datos/permutaciones200_2hsic.txt",delimiter="\t")
indep = np.loadtxt("./datos/permutacionesVSasint/HSIC-0.0-200version.txt",delimiter="\t")
x = np.array(range(len(solutions[0,:])))
x2 = np.array(range(len(solutions_as[0,:])))
for i in range(len(titles)):
	print(int(i/4),len(titles),i)
	ax[int(i/3),int(np.mod(i,3))].set_ylim((-0.1,1.1))
	ax[int(i/3),int(np.mod(i,3))].set_title(titles[i])
	ax[int(i/3),int(np.mod(i,3))].plot(x*3./10,solutions[i,:],label="Asymptotic")
	if (i != 11):
		ax[int(i/3),int(np.mod(i,3))].plot(x2*3./10,solutions_as[i,:],label="Real")
	else:
		ax[int(i/3),int(np.mod(i,3))].plot(np.linspace(0,3,5),indep[1,:],label="Real")
	ax[int(i/3),int(np.mod(i,3))].legend(loc='best')

figure,ax = plt.subplots(4,3, sharey=True)
figure.suptitle("RDC")
solutions = np.loadtxt("./datos/exp1Asint/RDC.txt",delimiter="\t")
solutions_as = np.loadtxt("./datos/permutaciones200_2rdc.txt",delimiter="\t")
indep = np.loadtxt("./datos/permutacionesVSasint/RDC-0.0-200version.txt",delimiter="\t")
x = np.array(range(len(solutions[0,:])))
x2 = np.array(range(len(solutions_as[0,:])))
for i in range(len(titles)):
	print(int(i/4),len(titles),i)
	ax[int(i/3),int(np.mod(i,3))].set_ylim((-0.1,1.1))
	ax[int(i/3),int(np.mod(i,3))].set_title(titles[i])
	ax[int(i/3),int(np.mod(i,3))].plot(x*3./10,solutions[i,:],label="Asymptotic")
	if (i != 11):
		ax[int(i/3),int(np.mod(i,3))].plot(x2*3./10,solutions_as[i,:],label="Real")
	else:
		ax[int(i/3),int(np.mod(i,3))].plot(np.linspace(0,3,5),indep[1,:],label="Real")
	ax[int(i/3),int(np.mod(i,3))].legend(loc='best')

plt.show()