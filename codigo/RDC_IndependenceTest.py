import numpy as np
from scipy.stats import chi2
from IndependenceTest import *

class RDC_IndependenceTest(IndependenceTest):
	def __init__(sef,filas,columnas,titulos):
		super().__init__("RDC",filas,columnas,titulos)
	def test(self,x,y,alpha,statistic = False):
		ros,ks = rdc(x,y)
		lx = len(x)
		count = 0.0
		aux = 1-np.power(ros,2)
		statistic_1 = ((2*ks+3)/2-lx)*np.log(np.prod(aux))
		pvalue = (1- chi2.cdf(statistic_1,df=ks**2))
		if pvalue < alpha:
			if(statistic):
				return[1,statistic_1]
			return 1
		else:
			if(statistic):
				return[0,statistic_1]
			return 0 

	def generate_statistic(self,x,y):
		ros,ks = rdc(x,y)
		lx = len(x)
		count = 0.0
		aux = 1-np.power(ros,2)
		statistic = ((2*ks+3)/2-lx)*np.log(np.prod(aux))
		return statistic


def ecdf(x):
	xs = np.sort(x)
	ys = np.arange(1,len(x)+1)/float(len(x))
	y  = np.ones(len(x))
	xs = xs.tolist()
	for i in range(len(ys)):
		y[i] = ys[xs.index(x[i])]
	return y
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
		return values,np.median(ks)
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
	[statistic,k] = cancor(fX,fY,k)
	return statistic,k
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