import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from scipy.stats import rankdata, chi2
from numpy.random import permutation
import time
class RDCtester:
	def __init__(self,n,b=100,nr=10):
		self.b = b
		self.reps = nr
		start_time = time.time()
		self.valores = resampling(n,b)
		elapsed_time = time.time() - start_time
		print("RDC:",elapsed_time)
		self.valores.sort()
	   # print(self.valores)
		self.times = []
	def computePvalue(self,rdcs):
		"""
		pn = 1./(self.b+1)
		count = 0.
		for i in range(self.b):
			if (dhsic <= self.valores[i]):
				count +=1.0
		pn += count/(self.b+1)
		#print(pn)
		"""
		l = np.zeros(len(rdcs))
		for i in range(len(rdcs)):
			l[i] = 1- sum(self.valores < rdcs[i])/(self.b*5)
		#print("pn",pn)
		return l

	def TestRDC(self,x,y, ruido):
		power = 0
		n = len(x)
		for i in range(self.reps):
			aux = y + np.random.normal(0,ruido,len(x))
			aux = (aux -np.mean(aux))/np.std(aux)
			start_time = time.time()
			estadistico,ks = rdc(x,aux,n=self.reps)
			ListPval = self.computePvalue(estadistico)
			elapsed_time = time.time() - start_time
			self.times.append(elapsed_time*1./len(estadistico))
			for pval in ListPval:
				if(pval < 0.01):
					power += 1./(self.reps*len(ListPval))
		return power

def resampling(n,b):
	x = np.random.normal(0,1, n)
	y = np.random.normal(0,1,n)
	valores = []
	for i in range(b):
		perm = permutation(n)
		e,k = rdc(x,y)
		valores.append(e)
	return valores

def ecdf(x):
	xs = np.sort(x)
	ys = np.arange(1,len(x)+1)/float(len(x))
	y  = np.ones(len(x))
	xs = xs.tolist()
	for i in range(len(ys)):
		y[i] = ys[xs.index(x[i])]
	return y
def tau(x):
	return 1/(1+ np.exp(-x))
def norm(k,s=1):
	ret =  np.random.normal(0,s,k)
	return ret
def rdc(x,y,k=10,s=0.2,n=1):
	if n > 1:
		values = []
		ks = []
		for i in range(n):
			#print(i,"   ",end=" ")
			try:
				ro,ko = rdc(x, y, k, s, 1)
				values.append(ro)
				ks.append(ko)
				#print(ro, end ="\r")
			except np.linalg.linalg.LinAlgError: pass
		return values,ks
	#print(x)
	lx = len(x)
	ly = len(y)
	x = np.concatenate((ecdf(x).reshape(-1,1),
		np.ones(lx).reshape(-1,1)),axis=1)
	#print(x)
	y = np.concatenate((ecdf(y).reshape(-1,1),
		np.ones(ly).reshape(-1,1)),axis=1)
	
	nx = x.shape[1]
	ny = y.shape[1]
	#print(nx,k)
	wx =  np.random.normal(0,s,nx*k).reshape(nx,k)
	wy =  np.random.normal(0,s,ny*k).reshape(ny,k)
	wxs = np.matmul(x,wx)
	wys = np.matmul(y,wy)
	fX = np.sin(wxs)
	fY = np.sin(wys)
	[res,k] = cancor(fX,fY,k)
	return res,k
	wxs = np.concatenate((np.cos(wxs),np.sin(wxs)),axis=1)
	wys = np.concatenate((np.cos(wys),np.sin(wys)),axis=1)
	#d = rcancor(wxs,wys)
	#print("mio",d)
	cca = CCA(n_components=1)
	cca.fit(wxs,wys)

	xc,yc = cca.transform(wxs,wys)
	#result1 = cca.score(xc,yc)
	result2 = np.corrcoef(xc.T,yc.T)[0,1]
	#print(result1,result2)
	return result2, k
def cancor(x,y,k):
	C = np.cov(np.hstack([x, y]).T)

	# Due to numerical issues, if k is too large,
	# then rank(fX) < k or rank(fY) < k, so we need
	# to find the largest k such that the eigenvalues
	# (canonical correlations) are real-valued

	k0 = k
	lb = 1
	ub = k
	while True:
		k = int(k)
		#print(k)

		# Compute canonical correlations
		Cxx = C[:k,:k]
		Cyy = C[k0:k0+k, k0:k0+k]
		Cxy = C[:k, k0:k0+k]
		Cyx = C[k0:k0+k, :k]

		eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.inv(Cxx), Cxy),
										np.dot(np.linalg.inv(Cyy), Cyx)))

		# Binary search if k is too large
		if not (np.all(np.isreal(eigs)) and
				0 <= np.min(eigs) and
				np.max(eigs) <= 1):
			ub -= 1
			k = (ub + lb) / 2
			continue
		if lb == ub: break
		lb = k
		if ub == lb + 1:
			k = ub
		else:
			k = (ub + lb) / 2

	return np.sqrt(eigs),k
def testRDC(x,y,ruido,k=7,n=5):
	#Under H0: two sets are uncorrelated:
	#our statistic will follow a xisquared distribution
	ruido =ruido/30
	for _ in range(n):
		aux = y + np.random.normal(0,ruido,len(x))
		aux = (aux -np.mean(aux))/np.std(aux)
		start_time = time.time()
		ros,ks = rdc(x,aux,k,n=n)
		count = 0.0
		for i in range(len(ros)):
			if(ros[i] <1):
				statistic = ((2*ks[i] +3)/2 - len(ros)) * np.log(1.0-ros[i]**2)
				p = (1- chi2.cdf(statistic,df=ks[i]**2))
				#print (p,end="\r")
			else:
				p = 0
			if p < 0.05:
				count +=1.0

	power = count* 1./len(ros)
	elapsed_time = time.time() - start_time
	times = elapsed_time* 1./len(ros)
	return power,times
def testRDC1Var(x,y,k=10):
	ros,ks = rdc(x,y)
	lx = len(x)
	count = 0.0
	aux = 1-ros**2
	statistic = ((2*k+3)/2-lx)*np.log(np.prod(aux))
	p = (1- chi2.cdf(statistic,df=ks**2))
	return p		

"""
n = 100
#test = RDCtester(n)
x = np.random.rand(n)
y = x**2
print("Res0\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res1\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res2\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res3\n",testRDC1Var(x,y))

y = (x-0.5)**2
print("Res0\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res1\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res2\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res3\n",testRDC1Var(x,y))

y = np.random.rand(n)
print("Res0\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res1\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res2\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res3\n",testRDC1Var(x,y))
#Menor que 0.05 no son indeps
#estadistico = HSIC_U_statistic_test(x,y)
#r = HSIC_V_statistic(kernelGausiano(x),kernelGausiano(y))
#print(r)
x = np.random.normal(0,1, n)
y = np.random.normal(0,1,n)
print("Res0\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res1\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res2\n",testRDC1Var(x,y))
y += np.random.normal(0,1,n)
print("Res3\n",testRDC1Var(x,y))
"""