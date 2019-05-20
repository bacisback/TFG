import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile,join
import pylab
import statsmodels.api as sm 
tests = ["RDC","HSIC","DCOV"]
onlfiles = [f for f in listdir("./datos")]
o = [s for s in onlfiles if ("permutaciones200_2" in s)]
o = [s for s in o if ("dcov" in s or "rdc" in s or "hsic" in s)]
titles = ["lineal","Parabolic","Cubic","Sin(4pix)","Sin(16pix)","fourth root","circle","step","xsin(x)","logarithm","gausian"]
figure,ax = plt.subplots(4,3, sharey=True)
figure.suptitle("Power")
cm = pylab.get_cmap('gist_rainbow')
for j,f in enumerate(o):
	solutions = np.loadtxt("./datos/"+f,delimiter="\t")
	if "dcov" in f:
		solutions = solutions * 100
	x = range(len(solutions[0,:]))
	print(f)
	for i in range(len(titles)):
		
		ax[int(i/3),int(np.mod(i,3))].set_ylim((-0.1,1.1))
		ax[int(i/3),int(np.mod(i,3))].set_title(titles[i])
		lowess = sm.nonparametric.lowess(solutions[i,:], x, frac=0.1) 
		ax[int(i/3),int(np.mod(i,3))].plot(lowess[:,0],lowess[:,1],color = cm(1.*j/3))
		if i == len(titles) -1:
			ax[int(i/3),int(np.mod(i,3))].plot(x,solutions[i,:],color = cm(1.*j/3),label=f[len("permutaciones200_2"):-3])
			ax[int(i/3),int(np.mod(i,3))].legend(loc='best')
for j,test in enumerate(tests):
	solutions = np.loadtxt("./datos/permutacionesVSasint/"+test+"-0.0-200version.txt",delimiter="\t")
	ax[3,2].plot(np.linspace(0,3,5),solutions[1,:],color = cm(1.*j/3),label = test)
ax[3,2].set_ylim((-0.1,1.1))
ax[3,2].set_title("2D gausian")
ax[3,2].legend(loc='best')
plt.show()
	
