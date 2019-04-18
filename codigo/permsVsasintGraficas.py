import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile,join
import pylab
onlfiles = [f for f in listdir("./datos/permutacionesVSasint")]
o = [s for s in onlfiles if ("RDC" in s)]
#o = onlfiles
noise = np.linspace(0,3,5)
for size in [50,100,150,200,500,1000]:
  actual = [s for s in o if ( "-"+str(size)+"." in s)]
  figure,ax = plt.subplots(1,5, sharey=True)
  figure.suptitle("RDC"+str(size))
  for f in actual:
#    title = f.replace("permutacionesVSasintHSIC-","").replace(str(size),"").replace("-.txt","")
    title = f.replace("RDC-","").replace("-"+str(size)+".","").replace("txt","")
    i = int(float(title)*1./0.25)
    print(i)
    solutions = np.loadtxt("./datos/permutacionesVSasint/"+f,delimiter="\t")
    ax[i].plot(noise,solutions[0,:],'b',label="asimptotic")
    ax[i].plot(noise,solutions[1,:],'r',label="real")
    ax[i].set_title(title)
    ax[i].legend(loc='best')
  plt.show()
