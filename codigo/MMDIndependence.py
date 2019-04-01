import numpy as np
import scipy as sp
from numpy import sqrt
from sklearn.metrics.pairwise import rbf_kernel
from numpy.random import permutation
import scipy 
from HSIC_IndependenceTest import *

def calculateKernel(patterns1,patterns2,deg = None):
  size = len(patterns1)
  if deg is None:
    xmed = patterns1
    G = xmed*xmed
    Q = np.repeat(G,size).reshape(size,size)
    R = Q.T
    dists = Q + R -2*xmed*xmed.T
    dists = dists - np.tril(dists)
    dists = dists.reshape(size*size,1)
    deg = np.sqrt(0.5*np.median(dists[dists>0]))

  G = patterns1*patterns1
  Q = np.repeat(G,size).reshape(size,size)
  R = Q.T
  if deg is None:
    dists = Q + R -2*xmed*xmed.T
    dists = dists - np.tril(dists)
    dists = dists.reshape(size*size,1)
    deg = np.sqrt(0.5*np.median(dists[dists>0]))
  H = patterns2*patterns2

  R = np.repeat(H,size).reshape(size,size).T

  H = Q+R-2*np.dot(patterns1,patterns2.T)

  H = np.exp(-H/2./(deg**2))
  
  return [H,deg]


#e = n(2/n sum E||yi-Z|| - E||Z-Z'|| - 1/n2 sum sum ||yi-yj||)
def energy_normality_test(x):

  x = x.flatten()
  y = np.sort(x)
  n = len(y)
  if(y[0] == y[-1]):
    return np.inf
  y = (y - np.mean(y))/np.std(y)
  K = np.linspace(-n,n,n)
  return 2 * (np.sum(2 * y * scipy.stats.norm(0, 1).cdf(y) + 2 * scipy.stats.norm(0, 1).pdf(y)) - n/np.sqrt(np.pi) - np.mean(K * y))


def h_normal(z):
  return 0.5*np.log(2*np.pi*np.exp(1)*np.var(z))

def h(z):
  euler_mascheroni = 0.57721566490153286060651209008240243104215933593992
  n = len(z)
  return np.sum(np.log(n*np.diff(np.sort(z))))/n + euler_mascheroni


  
def nGaussMMDenergy(x,y,rhoPoints=100):
  m = x.size
  if x.ndim == 1:
    x = x.reshape((-1,1))
  if y.ndim == 1:
    y = y.reshape((-1,1))
  rho = np.linspace(-1,1,rhoPoints)
  G = np.zeros(rhoPoints)
  Gperm = np.zeros(rhoPoints)
  MMD = np.zeros(rhoPoints)
  MMDperm = np.zeros(rhoPoints)
  E = np.zeros(rhoPoints)
  Eperm = np.zeros(rhoPoints)
  scaleE = np.zeros(rhoPoints)
  scaleMMD = np.zeros(rhoPoints)
  yperm = y[permutation(len(y))]
  
  Cxy = scipy.stats.pearsonr(x,y)[0][0]
  Cxyperm = 0
  for r in range(rhoPoints):
    r2 = rho[r]*rho[r]
    w = rho[r]*x + np.sqrt(1-r2)*y
    wperm = rho[r]*x + np.sqrt(1-r2)*yperm
    G[r] = np.log(2*np.pi*np.exp(1)*(1+2*rho[r]*np.sqrt(1-r2)*Cxy))/2 - h(w)
    Gperm[r] = np.log(2*np.pi*np.exp(1))/2 - h(wperm)
    MMD[r] = mmdStandardGaussian((w-np.mean(w))/np.std(w))
    MMDperm[r] = mmdStandardGaussian((w-np.mean(wperm))/np.std(wperm))
    E[r] = energy_normality_test(w)
    Eperm[r] = energy_normality_test(wperm)

    scaleE[r] = np.abs((h_normal(w) - h(w))/E[r])
    scaleMMD[r] = np.abs((h_normal(w) - h(w))/MMD[r])

  LD = -np.log(1-np.abs(Cxy))/2
  return {"diffG": (G-Gperm), "diffMMD" : (MMD-MMDperm), "diffE" :(E-Eperm), "LD":LD, "scaleE" : np.median(scaleE), "scaleMMD" : np.median(scaleMMD)}


def mmdStandardGaussian(sample):
  n = len(sample)
  [K,sigma] = calculateKernel(sample,sample)
  nonRandomTerm = sigma/np.sqrt(2+sigma*sigma)

  Y = np.sqrt(sigma*sigma / (1 + sigma*sigma)) * np.exp(-0.5 * np.power(sample,2) / (1 + sigma*sigma))

  Y -= np.diag(np.diag(Y))

  K -= np.diag(np.diag(K))

  randomTerm = 1 / (n * (n - 1)) *np.sum(K - Y - Y.T)

  MMD = randomTerm + nonRandomTerm
  return MMD

