import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from os import listdir
from os.path import isfile,join
import pylab
onlfiles = [f for f in listdir("./datos/histogramas")]
o = [s for s in onlfiles if ("DCOV" in s)]
#o = onlfiles
noise = np.linspace(0,3,5)
volume_of_experiment = 300
for size in [50,100,150,200,500,1000]:
  actual = [s for s in o if ( "-"+str(size)+"-" in s)]
  figure,ax = plt.subplots(1,5, sharey=True)
  figure.suptitle("DCOV"+" Size "+str(size))
  asimptotic = np.zeros((5,5))
  for f in actual:
#    title = f.replace("permutacionesVSasintDCOV-","").replace(str(size),"").replace("-.txt","")
    title = f.replace("DCOV-","").replace("-"+str(size),"").replace("txt","")
    corelacion,ruido = title.split("-")
    ruido = ruido[:-1]
    ruido_index = int(float(ruido)*1./0.75)
    cor_index = int(float(corelacion)*1./0.25)
    #print(i)
    #print(ruido_index,ruido,float(ruido)/0.75)
    hist = np.loadtxt("./datos/histogramas/"+f,delimiter="\t")
    #asimptotic[cor_index,ruido_index] = np.sum(hist>(chi2.ppf(0.95,df=1)-1))*1./volume_of_experiment
  for i,correlacion in enumerate(np.linspace(0,1,5)): 
    solutions = np.loadtxt("./datos/permutacionesVSasint/"+"DCOV-"+str(correlacion)+"-"+str(size)+"version.txt",delimiter="\t")
    ax[i].plot(noise,solutions[1,:],'r',label = "Real")
    ax[i].plot(noise,solutions[0,:],'b',label = "Asymptotic")
    ax[i].set_xlabel("Noise")
    ax[i].set_title(str(correlacion))
  ax[0].set_ylabel("Power")

  ax[4].legend(loc='best')
  #plt.show()
  plt.savefig("figuras/AsimptoticVsReal/DCOV"+str(size)+".png")
