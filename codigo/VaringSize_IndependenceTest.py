import pylab
import numpy as np
import matplotlib.pyplot as plt
import threading
import concurrent.futures
from funciones import *
from scipy.stats import chi2
from IndependenceTest import *
from HSIC_IndependenceTest import *
from RDC_IndependenceTest import *
from DCOV_IndependenceTest import *
from MMDIndependence import *
"""
Due to the ammount of shared code this tests have we will create a unique class which will preform all of the described tests
"""

class VaringSize_IndependenceTest:
  def __init__(self,sizes,reps,funciones,nsims,flag = False):
    ntests = 8
    self.funciones = funciones
    self.sizes = sizes
    self.solutions = np.zeros((ntests,len(funciones),len(sizes)))
    self.qGaus = np.zeros(len(sizes)) 
    self.qmax = np.zeros(len(sizes))
    self.qmean = np.zeros(len(sizes))
    self.qEmax = np.zeros(len(sizes))
    self.qEmean = np.zeros(len(sizes))
    self.qhsic = np.zeros(len(sizes))
    self.qrdc = np.zeros(len(sizes))
    self.qdcov = np.zeros(len(sizes))
    self.HSICt = HSIC_IndependenceTest(1,1,[])
    self.RDCt = RDC_IndependenceTest(1,1,[])
    self.DCOVt = DCOV_IndependenceTest(1,1,[])
    self.nsims=nsims
    if flag:
      return 
    for j,size in enumerate(sizes):
      gaussmean2 = np.zeros(reps)
      MMDmax2 = np.zeros(reps)
      MMDmean2 = np.zeros(reps)
      Emax2 = np.zeros(reps) 
      Emean2 = np.zeros(reps)
      hsic2 = np.zeros(reps)
      rdc2 = np.zeros(reps)
      dcov2 = np.zeros(reps)
      for i in range(reps):
        
        x,y = bivariate_gaussian(size,cov = np.eye(2))
        x = (x - np.mean(x))*1./np.std(x)
        y = (y - np.mean(y))*1./np.std(y)
        dataG = nGaussMMDenergy(x,y)
        hsic2[i] = self.HSICt.generate_statistic(x,y)
        alphaE = dataG["scaleE"]
        alphaMMD = dataG["scaleMMD"]
        gaussmean2[i] = np.mean(np.abs(dataG["diffG"])) + dataG["LD"]     
        MMDmax2[i] = np.max(np.abs(alphaMMD*dataG["diffMMD"])) + dataG["LD"]  
        MMDmean2[i] = np.mean(np.abs(alphaMMD*dataG["diffMMD"])) + dataG["LD"]  
        Emax2[i] = np.max(np.abs(alphaE*dataG["diffE"])) + dataG["LD"] 
        Emean2[i] = np.mean(np.abs(alphaE*dataG["diffE"])) + dataG["LD"]
        rdc2[i] = self.RDCt.generate_statistic(x,y)
        dcov2[i] = self.DCOVt.generate_statistic(x,y)
      
      self.qGaus[j] = np.percentile(gaussmean2,95) 
      self.qmax[j] = np.percentile(MMDmax2,95)
      self.qmean[j] = np.percentile(MMDmean2,95) 
      self.qEmax[j] = np.percentile(Emax2,95)  
      self.qEmean[j] = np.percentile(Emean2,95)
      self.qhsic[j] = np.percentile(hsic2,95)
      self.qrdc[j] = np.percentile(rdc2,95)
      self.qdcov[j] = np.percentile(dcov2,95)
    
    

  def test(self):
    for i, funcion in enumerate(self.funciones):
      with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.execute,i,funcion,j,size) for j,size in enumerate(self.sizes)}
            concurrent.futures.wait(futures)

  def execute(self,row,funcion,column,size):
    power = np.zeros(8)
    for _ in range(self.nsims):
      x,y = funcion(size)
      x = (x - np.mean(x))*1./np.std(x)
      y = (y - np.mean(y))*1./np.std(y)
      dataG = nGaussMMDenergy(x,y)
      alphaE = dataG["scaleE"]
      alphaMMD = dataG["scaleMMD"]
      gaussmean = np.mean(np.abs(dataG["diffG"])) + dataG["LD"]     
      MMDmax = np.max(np.abs(alphaMMD*dataG["diffMMD"])) + dataG["LD"]  
      MMDmean = np.mean(np.abs(alphaMMD*dataG["diffMMD"])) + dataG["LD"]  
      Emax = np.max(np.abs(alphaE*dataG["diffE"])) + dataG["LD"] 
      Emean = np.mean(np.abs(alphaE*dataG["diffE"])) + dataG["LD"] 
      hsic = self.HSICt.generate_statistic(x,y)
      rdc_statistic = self.RDCt.generate_statistic(x,y)
      [asimptotic,dcov_statistic]=self.DCOVt.test(x,y,0.05,True)
      power[0] += int(gaussmean > self.qGaus[column])
      power[1] += int(MMDmax > self.qmax[column])
      power[2] += int(MMDmean > self.qmean[column])
      power[3] += int(Emax > self.qEmax[column])
      power[4] += int(Emean > self.qEmean[column])
      power[5] += int(hsic > self.qhsic[column])
      power[6] += int(rdc_statistic > self.qrdc[column])
      power[7] += int(dcov_statistic > self.qdcov[column])
    power *= 1./self.nsims
    self.solutions[:,row,column] = power
    print(self.solutions[:,row,column])

  def print(self,text):
    test = ["gaussmean","MMDmax","MMDmean","Emax","Emean","hsic","rdc","dcov"]
    titulos = ["bivariate gaussian","Gaussian multiply uniform", "Mixture of 3 gaussians","Gaussian multiplicative noise"]
    cm = pylab.get_cmap('gist_rainbow')
    fig,ax = plt.subplots(2,2,sharey=True)
    for j,matrix in enumerate(self.solutions):
      for i in range(len(matrix)):
        ax[int(i/2),np.mod(i,2)].plot(self.sizes,matrix[i,:],color = cm(1.*j/8),label = test[j])

        ax[int(i/2),np.mod(i,2)].set_title(titulos[i])
        ax[int(i/2),np.mod(i,2)].legend(loc='best')
        ax[int(i/2),np.mod(i,2)].set_ylim(-0.01,1.01)
    plt.show()
    for j,matrix in enumerate(self.solutions):
      np.savetxt(text+test[j]+"500.txt",matrix,delimiter="\t")

funciones = [bivariate_gaussian,gaussian_multiply_uniform,mixture_3_gaussians,gaussian_multiplicative_noise]
ninit = 10
nend = 500
n = 7
t = VaringSize_IndependenceTest(np.linspace(ninit,nend,n,dtype=int),500,funciones,500)
#t.execute(1,bivariate_gaussian,1,200)
t.test()
t.print("datos/testPerms/")

#t = BasedOnGauss_IndependenceTest(200,100,[twod_gaussian],np.linspace(0,3,10),200)
#t.execute_independent(5)
#t.test_independent_and_varing_size()
