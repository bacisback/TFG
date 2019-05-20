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

class BasedOnGauss_IndependenceTest:
  def __init__(self,size,reps,funciones,steps,nsims,flag =False):
    
    self.funciones = funciones
    self.size = size
    self.steps = steps
    self.solutions = np.zeros((8,len(funciones),len(steps)))
    gaussmean2 = np.zeros(reps)
    MMDmax2 = np.zeros(reps)
    MMDmean2 = np.zeros(reps)
    Emax2 = np.zeros(reps) 
    Emean2 = np.zeros(reps)
    hsic2 = np.zeros(reps)
    rdc2 = np.zeros(reps)
    self.HSICt = HSIC_IndependenceTest(1,1,[])
    self.RDCt = RDC_IndependenceTest(1,1,[])
    self.nsims=nsims
    if flag:
      self.qGaus = np.percentile(gaussmean2,95) 
      self.qmax = np.percentile(MMDmax2,95)
      self.qmean = np.percentile(MMDmean2,95) 
      self.qEmax = np.percentile(Emax2,95)  
      self.qEmean = np.percentile(Emean2,95)
      self.qhsic = np.percentile(hsic2,95)
      self.qrdc = np.percentile(rdc2,95)
      return 
    for i in range(reps):
      x = np.random.rand(size)
      y = np.random.rand(size)
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
      

    self.qGaus = np.percentile(gaussmean2,95) 
    self.qmax = np.percentile(MMDmax2,95)
    self.qmean = np.percentile(MMDmean2,95) 
    self.qEmax = np.percentile(Emax2,95)  
    self.qEmean = np.percentile(Emean2,95)
    self.qhsic = np.percentile(hsic2,95)
    self.qrdc = np.percentile(rdc2,95)
    print(self.qGaus,self.qmax,self.qmean,self.qEmax,self.qEmean,self.qhsic,self.qrdc)
    

  def test(self):
    for i, funcion in enumerate(self.funciones):
      with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.execute,i,funcion,j) for j in range(len(self.steps))}
            concurrent.futures.wait(futures)
      print(i*1./len(self.funciones))
      print(self.solutions[:,i,:])
  def test_independent_and_varing_size(self):
    self.solution = np.zeros((8,len(self.steps)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
          futures = {executor.submit(self.execute_independent,j) for j in range(len(self.steps))}
          concurrent.futures.wait(futures)
    np.savetxt("datos/permutaciones200_2_independientes.txt",self.solution)

  def execute_independent(self,column):
    power = np.zeros(8)
    for _ in range(self.nsims):
      x,y = bivariate_gaussian(self.size,cov = np.eye(2))
      y = y + np.random.normal(0,self.steps[column],self.size)
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
      [Sdcov,Sdcor,pvalue] = dcov(x,y,0.05,R = self.nsims)
      if pvalue < 0.05:
        power[7] +=0.01
      power[0] += int(gaussmean > self.qGaus)
      power[1] += int(MMDmax > self.qmax)
      power[2] += int(MMDmean > self.qmean)
      power[3] += int(Emax > self.qEmax)
      power[4] += int(Emean > self.qEmean)
      power[5] += int(hsic > self.qhsic)
      power[6] += int(rdc_statistic > self.qrdc)
    power *= 1./self.nsims
    self.solutions[:,column] = power

  def execute(self,row,funcion,column):
    power = np.zeros(8)
    for _ in range(self.nsims):
      x = np.random.rand(self.size)
      z = funcion(x)
      y = np.array([j for j in z]) + np.random.normal(0,self.steps[column],self.size)
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
      [Sdcov,Sdcor,pvalue] = dcov(x,y,0.05,R = self.nsims)
      if pvalue < 0.05:
        power[7] +=0.01
      power[0] += int(gaussmean > self.qGaus)
      power[1] += int(MMDmax > self.qmax)
      power[2] += int(MMDmean > self.qmean)
      power[3] += int(Emax > self.qEmax)
      power[4] += int(Emean > self.qEmean)
      power[5] += int(hsic > self.qhsic)
      power[6] += int(rdc_statistic > self.qrdc)
    power *= 1./self.nsims
    self.solution[:,row,column] = power

  def print(self,text):
    test = ["gaussmean","MMDmax","MMDmean","Emax","Emean","hsic","rdc","dcov"]
    titulos = ["Linear","Parabolic","Cubic","Sin1","Sin2","root4","circle","step","xsin","logarithm","gausian"]
    cm = pylab.get_cmap('gist_rainbow')
    fig,ax = plt.subplots(4,3,sharey=True)
    for j,matrix in enumerate(self.solutions):
      for i in range(len(matrix)):
        ax[int(i/3),np.mod(i,3)].plot(self.steps,matrix[i,:],color = cm(1.*j/8),label = test[j])

        ax[int(i/3),np.mod(i,3)].set_title(titulos[i])
        ax[int(i/3),np.mod(i,3)].legend(loc='best')
        ax[int(i/3),np.mod(i,3)].set_ylim(-0.01,1.01)
    plt.show()
    for j,matrix in enumerate(self.solutions):
      np.savetxt(text+test[j]+".txt",matrix,delimiter="\t")

t = BasedOnGauss_IndependenceTest(200,100,[Linear,Parabolic,Cubic,Sin1,Sin2,root4,circle,step,xsin,logarithm,gausian],np.linspace(0,3,10),100)
t.test()
t.print("datos/permutaciones200_2")

#t = BasedOnGauss_IndependenceTest(200,100,[twod_gaussian],np.linspace(0,3,10),200)
#t.execute_independent(5)
#t.test_independent_and_varing_size()
