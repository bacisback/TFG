from IndependenceTest import *
from HSIC_IndependenceTest import *
from RDC_IndependenceTest import *
from DCOV_IndependenceTest import *
from MMDIndependence import *
import numpy as np
import matplotlib.pyplot as plt
import threading
import concurrent.futures
from funciones import *



nsims1 = nsims1
functions = [bivariate_gaussian,gaussian_multiply_uniform,mixture_3_gaussians,gaussian_multiplicative_noise]
for s in size
  gaussmean2 = np.zeros(nsims1)
  MMDmax2 = np.zeros(nsims1)
  MMDmean2 = np.zeros(nsims1)
  Emax2 = np.zeros(nsims1) 
  Emean2 = np.zeros(nsims1)
  hsic2 = np.zeros(nsims1)
  rdc2 = np.zeros(nsims1)
  dcov2 = np.zeros(nsims1)

  for i in range(nsims1):
    x = np.random.rand(500)
    y = np.random.rand(500)
    dataG = nGaussMMDenergy(x,y)
    [hsic2[i],_] = HSIC_test_gamma(x,y,0.05)
    alphaE = dataG["scaleE"]
    alphaMMD = dataG["scaleMMD"]
    gaussmean2[i] = np.mean(np.abs(dataG["diffG"])) + dataG["LD"]     
    MMDmax2[i] = np.max(np.abs(alphaMMD*dataG["diffMMD"])) + dataG["LD"]  
    MMDmean2[i] = np.mean(np.abs(alphaMMD*dataG["diffMMD"])) + dataG["LD"]  
    Emax2[i] = np.max(np.abs(alphaE*dataG["diffE"])) + dataG["LD"] 
    Emean2[i] = np.mean(np.abs(alphaE*dataG["diffE"])) + dataG["LD"]
  qGaus = np.percentile(gaussmean2,95) 
  qmax = np.percentile(MMDmax2,95)
  qmean = np.percentile(MMDmean2,95) 
  qEmax = np.percentile(Emax2,95)  
  qEmean = np.percentile(Emean2,95)
  qhsic = np.percentile(hsic2,95)
  qrdc = np.percentile(rdc,95)
  qdcov = np.percentile(dcov,95)
  power = np.zeros(6)

  for i in range(nsims2):
    x = np.random.rand(500)
    y = np.power(x,2) + np.random.normal(0,3,500)
    dataG = nGaussMMDenergy(x,y)
    alphaE = dataG["scaleE"]
    alphaMMD = dataG["scaleMMD"]
    gaussmean = np.mean(np.abs(dataG["diffG"])) + dataG["LD"]     
    MMDmax = np.max(np.abs(alphaMMD*dataG["diffMMD"])) + dataG["LD"]  
    MMDmean = np.mean(np.abs(alphaMMD*dataG["diffMMD"])) + dataG["LD"]  
    Emax = np.max(np.abs(alphaE*dataG["diffE"])) + dataG["LD"] 
    Emean = np.mean(np.abs(alphaE*dataG["diffE"])) + dataG["LD"] 
    hsic,_ = HSIC_test_gamma(x,y,0.05)
    #print(gaussmean,MMDmax,MMDmean,Emax,Emean)
    power[0] += int(gaussmean > qGaus)
    power[1] += int(MMDmax > qmax)
    power[2] += int(MMDmean > qmean)
    power[3] += int(Emax > qEmax)
    power[4] += int(Emean > qEmean)
    power[5] += int(hsic > qhsic)
  print(power*1./nsims2)
