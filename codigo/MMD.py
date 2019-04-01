import numpy as np
import scipy as sp
from numpy import sqrt
from sklearn.metrics.pairwise import rbf_kernel
from numpy.random import permutation
from scipy.stats import gamma
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

def rbf_dot(patterns1,patterns2,deg):
	size1 = len(patterns1)

	G = patterns1*patterns1
	Q = np.repeat(G,size1).reshape(size1,size1)
	
	H = patterns2*patterns2

	R = np.repeat(H,size1).reshape(size1,size1).T

	H = Q+R-2*np.dot(patterns1,patterns2.T)

	H = np.exp(-H/2./(deg**2))
	
	return H

def fit_gamma(x,y,alpha=0.05,params=[-1,-1,-1]):
	m = len(x)
	m1 = (m-1)*m
	x = x.reshape((-1,1))
	y = y.reshape((-1,1))
	"""
	Sacamos los kernels de X e Y
	"""
	if params[0] == -1:
		size1 = m
		if size1 > 100:
			xmed = x[0:100]
			size1 = 100
		else:
			xmed = x
		G = xmed*xmed
		Q = np.repeat(G,size1).reshape(size1,size1)
		R = Q.T
		dists = Q + R -2*xmed*xmed.T
		dists = dists - np.tril(dists)
		dists = dists.reshape(size1*size1,1)
		params[0] = np.sqrt(0.5*np.median(dists[dists>0]))
	if params[1] == -1:
		size1 = m
		if size1 > 100:
			ymed = y[0:100]
			size1 = 100
		else:
			ymed = y
		G = ymed*ymed
		Q = np.repeat(G,size1).reshape(size1,size1)
		R = Q.T
		dists = Q + R -2*ymed*ymed.T
		dists = dists - np.tril(dists)
		dists = dists.reshape(size1*size1,1)
		params[1] = np.sqrt(0.5*np.median(dists[dists>0]))

	if params[2] == -1:
		size1 = m
		if size1 > 100:
			xmed = x[0:100]
			ymed = y[0:100]
			size1 = 100
		else:
			xmed = x
			ymed = y
		G = xmed*xmed
		Q = np.repeat(G,size1).reshape(size1,size1)
		R = ymed*ymed
		R = np.repeat(R,size1).reshape(size1,size1)
		dists = Q + R -2*xmed*ymed.T
		dists = dists - np.tril(dists)
		dists = dists.reshape(size1*size1,1)
		params[1] = np.sqrt(0.5*np.median(dists[dists>0]))

	Kxx = rbf_dot(x,x,params[0])
	#Kxx -= np.eye(m)
	Kyy = rbf_dot(y,y,params[1])
	#Kyy -= np.eye(m)
	Kxy = rbf_dot(x,y,params[2])
	#Kxy -= np.eye(m)
	Kyy -= np.eye(m)
	Kxx -= np.eye(m)
	Kxy -= np.diag(np.diag(Kxy))
	H = Kxx + Kyy -2* Kxy
	
	
	#stat = np.sum(Haux -np.diag(np.diag(Haux)))/m1
	stat = np.sum(H)/m1
	#meanMMD =2/m * ( 1  - 1/m*sum(diag(KL))  );
	meanMMD = 2./m * (1 - 1./m * (np.sum(np.diag(Kxy))))
	#varMMD=2/m/(m-1) * 1/m/(m-1) * sum(sum( (K+L - KL - KL').^2 ));
	varMMD = 2./m/(m-1) * 1./m/(m-1) * np.sum(np.power(H,2))

	

	al = np.power(meanMMD,2)*1./varMMD
	bet = varMMD*m*1./meanMMD

	thresh = 1- gamma.cdf(1-alpha,al,bet)

	
	return [stat,thresh]


n = 300

x = np.random.normal(0,1,n)
x = (x - np.mean(x))/np.std(x)
y = np.random.normal(0,1, n)
y = (y - np.mean(y))/np.std(y)
print(fit_gamma(x,y))

x = np.random.rand(n)

y = np.random.rand(n)
x = (x - np.mean(x))/np.std(x)
y = (y - np.mean(y))/np.std(y)

print(fit_gamma(x,y))