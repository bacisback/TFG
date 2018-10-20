#Generaremos una funcion para crear kernels gausianos:

from scipy.spatial.distance import cdist,pdist,squareform
from numpy import fill_diagonal
import numpy as np
from scipy.stats import norm as normaldist, gamma
from numpy.random import permutation
from math import factorial
def kernelGausiano(X,Y=None, sigma = 1.0):
	"""
	Como en un principio los nucleos solo los vamos a usar para calcular
	el de la distribucion no de una conjunta Y = None siempre
	"""
	n = X.ndim
	if n == 1:
		n = len(X)
		norma = np.zeros((n,n))
		for i in range(n):
			for j in range(i+1,n):
				#print(np.exp(-sigma**(-2)* (X[i] - X[j])**2))
				norma[i,j] = np.exp(-sigma**(-2)* (X[i] - X[j])**2)
				norma[j,i] = norma[i,j] 

		return norma
	else:
		if Y is None:
			Y = X
		
			norma = squareform(pdist(X, 'sqeuclidean'))
		else:
			norma =  cdist(X,Y,'sqeuclidean')
		return np.exp(-sigma**(-2)* norma)
def sigma_median_heuristic(X):
	n = X.ndim
	if n == 1:
		n = len(X)
		norma = np.zeros((n,n))
		for i in range(n):
			for j in range(i+1,n):
				#print(np.exp(-sigma**(-2)* (X[i] - X[j])**2))
				norma[i,j] = np.exp(-sigma**(-2)* (X[i] - X[j])**2)
				norma[j,i] = norma[i,j] 

	else:
		if Y is None:
			Y = X
		
			norma = squareform(pdist(X, 'sqeuclidean'))
		else:
			norma =  cdist(X,Y,'sqeuclidean')
	median_dist = np.median(norma[norma>0])
	sigma = median_dist/np.sqrt(2.)
	return sigma
def center_kernel_matrix(K,hkh=False):
	"""
    Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
    """
	n = np.shape(K)[0]
	H = np.eye(n) - (1./n)*np.ones((n,1)).dot(np.ones((1,n)))
	if hkh:
		return H.dot(K.dot(H))
	return 1.0 / n * K.dot(H)
def HSIC_U_statistic(Kx,Ky):

	"""Como no sabemos si el nucleo al coger 
	la distancia ha tenido errores de redondeo y las diagonal 
	es todo 0 ya que vamos a implementar un nucleo gausiano
	kij = exp(-sigma^-2 norm(xi-xj)^2) xi -xj = 0 si i = j
	
	"""
	m = np.shape(Kx)[0] #Suponemos Kx y Ky de mismas dimensiones
	#print(Kx)
	fill_diagonal(Kx,0.)
	fill_diagonal(Ky,0.)
	#print(Kx,Ky)
	K = np.dot(Kx,Ky)
	#print(K)
	#print(K)
	t1 = np.trace(K)/float(m*(m-1))
	t2 = np.sum(Kx)*np.sum(Ky)/float(m*(m-1.)*(m-2.)*(m-3.))
	t3 = np.sum(K)/float(m*(m-1.)*(m-2.))
	return t1+t2-2.*t3
def HSIC_V_statistic(KX,KY):
	Kxc = center_kernel_matrix(KX)
	Kyc = center_kernel_matrix(KY)
	return np.trace(Kxc*Kyc)/np.power(np.shape(KX)[0],2)

def null_samplesHsic(Kx,Ky):
	null_samples=np.zeros(1)
	perm = permutation(np.shape(Ky)[0])
	Kpp = Ky[perm,:][:,perm]
	null_samples[0]=HSIC_U_statistic(Kx,Kpp)
	print(null_samples)
	return null_samples

def HSIC_U_statistic_test(x,y,blocksize = 5, nblocks = 100):
	Btest = np.zeros(nblocks)
	for i in range(nblocks):
		indx1 = i*blocksize
		indx2 = indx1+blocksize
		kx = kernelGausiano(x[indx1:indx2])
		ky = kernelGausiano(y[indx1:indx2])
		Btest[i]= HSIC_U_statistic(kx,ky)
	Btest_Statistic = sum(Btest)/float(nblocks)
	Btest_nullVar = np.var(Btest)
	z_score = np.sqrt(blocksize*nblocks)*Btest_Statistic/np.sqrt(Btest_nullVar)
	print("perm-pv",normaldist.sf(z_score))
	kx = kernelGausiano(x)
	ky = kernelGausiano(y)
	ft = HSIC_U_statistic(kx,ky)
	st =  HSIC_U_statistic(kx,kx)*  HSIC_U_statistic(ky,ky)
	r = ft/(np.sqrt(st))
	#test normaldist.sf(ustatistic) < alpha?
	return r
def Calcular_Esperanza_Var(x,y):
	# Btest = np.zeros(n)
	# for i in range(n):
	# 	perm = permutation(blocksize)
	# 	kx = kernelGausiano(x[perm])
	# 	ky = kernelGausiano(y[perm])
	# 	Btest[i] = HSIC_U_statistic(kx,ky)
	# return [np.mean(Btest),(np.std(Btest)*n**3./2)**2]
	Kx = kernelGausiano(x)
	Ky = kernelGausiano(y)
	m = np.shape(Kx)[0]
	mux = sum(sum(Kx))*(1./(m*(m-1)))
	muy = sum(sum(Ky))*(1./(m*(m-1)))
	mean = (1./m)*(1+ mux*muy - mux-muy)
	K = center_kernel_matrix(Kx,True)
	L = center_kernel_matrix(Ky,True)
	B = np.matrix(K*L)
	B = np.power(np.matrix(B),-2)
	onesT = np.ones((1,m))
	ones = np.ones((m,1))
	var = onesT.dot((B-np.trace(B)).dot(ones))
	return [mean,var]

def gamma_valor(x,a,b):
	return np.power(x,a-1)*np.exp(-x/b)/(np.power(b,a)*factorial(np.floor(a)))
n = 500
#np.random.seed(1)
x = np.random.rand(n)
y = np.random.normal(0,1, n)
[mu,sigma] = Calcular_Esperanza_Var(x,y)
print(mu,sigma)
estadistico = HSIC_U_statistic_test(x,y)
print("estadistico",normaldist.sf(estadistico))
r = HSIC_V_statistic(kernelGausiano(x),kernelGausiano(y))
print("Vstatistic2",r)

a = mu**2/sigma
#print(a)
b = n*sigma/mu

pvalue = gamma.pdf(n*r,a,b)
print(pvalue)
y = np.copy(x)

estadistico = HSIC_U_statistic_test(x,y)
print("estadistico",normaldist.sf(estadistico))
r = HSIC_V_statistic(kernelGausiano(x),kernelGausiano(y))
print("Vstatistic2",r)
print("pvalor v",normaldist.sf(r))
[mu,sigma] = Calcular_Esperanza_Var(x,y)
print(mu,sigma)
a = mu**2/sigma
#print(a)
b = n*sigma/mu

pvalue = gamma.pdf(n*r,a,b)
print(pvalue)