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

class VaringRotationAngle_IndependenceTest:
  def __init__(self,size,reps,nsims,angles,flag = False):
    self.ntests = 8
    self.angles = angles
    self.size = size
    self.solutions = np.zeros((self.ntests,len(angles)))
    self.qGaus = 0 
    self.qmax = 0
    self.qmean = 0
    self.qEmax = 0
    self.qEmean = 0
    self.qhsic = 0
    self.qrdc = 0
    self.qdcov = 0
    self.HSICt = HSIC_IndependenceTest(1,1,[])
    self.RDCt = RDC_IndependenceTest(1,1,[])
    self.DCOVt = DCOV_IndependenceTest(1,1,[])
    self.nsims=nsims
    if flag:
      return 
    
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
      
    self.qGaus = np.percentile(gaussmean2,95) 
    self.qmax = np.percentile(MMDmax2,95)
    self.qmean = np.percentile(MMDmean2,95) 
    self.qEmax = np.percentile(Emax2,95)  
    self.qEmean = np.percentile(Emean2,95)
    self.qhsic = np.percentile(hsic2,95)
    self.qrdc = np.percentile(rdc2,95)
    self.qdcov = np.percentile(dcov2,95)
    
    

  def test(self):
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
          futures = {executor.submit(self.execute,j,angle) for j,angle in enumerate(self.angles)}
          concurrent.futures.wait(futures)

  def execute(self,column,angle):
    power = np.zeros(8)
    for _ in range(self.nsims):
      x,y = rotation(self.size,angle*np.pi)
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
      power[0] += int(gaussmean > self.qGaus)
      power[1] += int(MMDmax > self.qmax)
      power[2] += int(MMDmean > self.qmean)
      power[3] += int(Emax > self.qEmax)
      power[4] += int(Emean > self.qEmean)
      power[5] += int(hsic > self.qhsic)
      power[6] += int(rdc_statistic > self.qrdc)
      power[7] += int(dcov_statistic > self.qdcov)
    power *= 1./self.nsims
    self.solutions[:,column] = power
    print(self.solutions[:,column])

  def print(self,text):
    test = ["gaussmean","MMDmax","MMDmean","Emax","Emean","hsic","rdc","dcov"]
    cm = pylab.get_cmap('gist_rainbow')
    for j in range(self.ntests):
      
      plt.plot(self.angles,self.solutions[j,:],color = cm(1.*j/8),label = test[j])
      plt.title("Rotation Experiment")
      plt.xlabel('Rotation angle times pi')
      plt.ylabel('Power')
      plt.legend(loc='best')
      plt.ylim(-0.01,1.01)
    plt.show()
    np.savetxt(text+"experiment100samples.txt",self.solutions,delimiter="\t")

ninit = 0
nend = 2
n = 100
t = VaringRotationAngle_IndependenceTest(200,500,500,np.linspace(ninit,nend,n))
#t.execute(1,0.5)
t.test()
t.print("datos/rotations/")

#t = BasedOnGauss_IndependenceTest(200,100,[twod_gaussian],np.linspace(0,3,10),200)
#t.execute_independent(5)
#t.test_independent_and_varing_size()
