import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile,join
import pylab
onlfiles = [f for f in listdir("./datos")]
o = [s for s in onlfiles if ("permutaciones200_2" in s)]
titles = ["lineal","Parabolic","Cubic","Sin(4pix)","Sin(16pix)","fourth root","circle","step","xsin(x)","logarithm","gausian"]
figure,ax = plt.subplots(4,3, sharey=True)
figure.suptitle("Power")
cm = pylab.get_cmap('gist_rainbow')
for j,f in enumerate(o):
	solutions = np.loadtxt("./datos/"+f,delimiter="\t")
	if "dcov" in f:
		solutions = solutions * 100
	x = range(len(solutions[0,:]))
	for i in range(len(titles)):
		print(int(i/4),len(titles),i)
		ax[int(i/3),int(np.mod(i,3))].set_ylim((-0.1,1.1))
		ax[int(i/3),int(np.mod(i,3))].set_title(titles[i])
		ax[int(i/3),int(np.mod(i,3))].plot(x,solutions[i,:],color = cm(1.*j/8))
		if i == len(titles) -1:
			ax[int(i/3),int(np.mod(i,3))].plot(x,solutions[i,:],color = cm(1.*j/8),label=f[len("permutaciones200_2"):-3])
			ax[int(i/3),int(np.mod(i,3))].legend(loc='best')
plt.show()
	
