import numpy as np
import scipy as sp
from numpy import sqrt
from sklearn.metrics.pairwise import rbf_kernel
from numpy.random import permutation

def kernelGausiano(X,Y=None, sigma = 1.0):
	"""
	Como en un principio los nucleos solo los vamos a usar para calcular
	el de la distribucion no de una conjunta Y = None siempre
	"""
	n = X.ndim
	if Y is None:
		Y = X
	#print(X)
	if n == 1:
		n = len(X)
		#print(n)
		norma = np.zeros((n,n))
		for i in range(n):
			for j in range(i+1,n):
				#print(np.exp(-sigma**(-2)* (X[i] - X[j])**2))
				norma[i,j] = np.exp(-sigma**(-2)* (X[i] - Y[j])**2)
				norma[j,i] = norma[i,j] 
				#print(norma[i,j])
		return norma
	else:
		norma =  cdist(X,Y,'sqeuclidean')
	return np.exp(-sigma**(-2)* norma)


def U_Statistic(x,y):
	m = len(x)
	K_XX = kernelGausiano(x)
	K_YY = kernelGausiano(y)
	K_XY = kernelGausiano(x,y)
	r = max([K_XX.max(),K_YY.max()])

	t1 = np.sum(np.sum(K_XX)) * 1./(m*(m-1))
	t2 = np.sum(np.sum(K_YY)) * 1./(m*(m-1))
	t3 = np.sum(np.sum(K_XY)) * 2./(m*(m-1))

	mmd2 = t1+t2-t3

	pvalue = 0.0
	for i in range(200):
		mmdAux = 0.0
		perm = permutation(m)
		yaux = y[perm]
		kaux = kernelGausiano(x,yaux)
		t3 = np.sum(np.sum(kaux)) * 2./(m*(m-1))
		mmdAux = t1+t2-t3
		if mmd2 < mmdAux:
			pvalue +=1
	pvalue = pvalue *1./200
	return mmd2,pvalue

def grbf(x1, x2, sigma = 1):
	'''Calculates the Gaussian radial base function kernel'''
	print(x1.shape)
	n, nfeatures = x1.shape
	m, mfeatures = x2.shape

	k1 = np.sum((x1.reshape((-1,1))*x1), 1)
	q = np.tile(k1, (m, 1)).transpose()
	del k1

	k2 = np.sum((x2.reshape((-1,1))*x2), 1)
	r = np.tile(k2.T, (n, 1))
	del k2

	h = q + r
	del q,r

	# The norm
	h = h - 2*np.dot(x1,x2.transpose())
	h = np.array(h, dtype=float)

	return np.exp(-1.*h/(2.*np.power(sigma,2)))

def tester_MMD(x,y,ruido):
	power = 0
	reps = 10
	n = len(x)
	for j in range(reps):
		aux = y + np.random.normal(0,ruido,len(x))
		aux = (aux -np.mean(aux))/np.std(aux)
		statistic,var = U_Statistic(x,aux)
		if(statistic > var):
			power += 1./reps
	return power

n = 500

x = np.random.rand(n)
x = (x - np.mean(x))/np.std(x)
y = np.random.normal(0,1, n)
y = (y - np.mean(y))/np.std(y)

statistic,var = U_Statistic(x,y)


#Reject Ho p=q if MMD > 2alfa*var/sqrt(m)
print(statistic )
print(var)

y = np.random.rand(n)
statistic,var = U_Statistic(x,y)
#Reject Ho p=q if MMD > 2alfa*var/sqrt(m)
print(statistic )
print(var)

