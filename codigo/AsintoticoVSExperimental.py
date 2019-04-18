import pylab
import numpy as np
import matplotlib.pyplot as plt
import threading
import concurrent.futures
from IndependenceTest import *
from HSIC_IndependenceTest import *
from RDC_IndependenceTest import *
from DCOV_IndependenceTest import *
def test(independenceObject):
    for size in [50,100,150,200,500,1000]:
        print(size)
        dist = np.zeros(500)
        for i in range(500):
            x, y = np.random.multivariate_normal(np.ones(2), np.eye(2), size).T
            dist[i] = independenceObject.generate_statistic(x,y)
        #print(dist)
        np.savetxt("datos/histogramas/"+str(independenceObject)+"-independent-"+str(size)+".txt",dist,delimiter="\t")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(execute,ro,size,independenceObject,np.percentile(dist,95)) for ro in np.linspace(0,1,5)}
            concurrent.futures.wait(futures)
        #plt.show()
def execute(ro,size,independenceObject,percentile):
    #print(ro,size)
    n = 300
    power_asimptotic = np.zeros(5)
    power_real = np.zeros(5)
    noise = np.linspace(0,3,5)
    statistics = np.zeros(n)
    for i in range(5):
        for j in range(n):
            x = np.random.normal(0,1,size)
            y = ro*x + np.sqrt(1-ro*ro)*np.random.normal(0,1,size) + np.random.normal(0,noise[i],size)
            x = (x - np.mean(x))*1./np.std(x)
            y = (y - np.mean(y))*1./np.std(y)
            [asimptotic,statistic]=independenceObject.test(x,y,0.05,True)
            statistics[j] = statistic
            power_asimptotic[i] += asimptotic*1./n
            power_real[i] += int(statistic>percentile)*1./n
        np.savetxt("datos/histogramas/"+str(independenceObject)+"-"+str(ro)+"-"+str(size)+"-"+str(noise[i])+".txt",statistics,delimiter="\t")

    
    np.savetxt("datos/permutacionesVSasint/"+str(independenceObject)+"-"+str(ro)+"-"+str(size)+"version.txt",np.stack((power_asimptotic,power_real)),delimiter="\t")

HSIC = DCOV_IndependenceTest(1,1,[])
#execute(1,500,HSIC,5)
test(HSIC)