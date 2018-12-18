#Generaremos una funcion para crear kernels gausianos:

from scipy.spatial.distance import cdist,pdist,squareform
from numpy import fill_diagonal
import numpy as np
from scipy.stats import norm as normaldist, gamma
from numpy.random import permutation
from math import factorial
import time
class HSIC:
	def __init__(self,n,b=500,nr=10):
		self.b = b
		self.reps = nr
		start_time = time.time()
		self.valores = resampling(n,b)
		elapsed_time = time.time() - start_time
		print(elapsed_time)
		self.valores.sort()
		self.times = []

	def computePvalue(self,dhsic):
		"""
		pn = 1./(self.b+1)
		count = 0.
		for i in range(self.b):
			if (dhsic <= self.valores[i]):
				count +=1.0
		pn += count/(self.b+1)
		#print(pn)
		"""
		pn = 1- sum(self.valores < dhsic)/self.b
		#print("pn",pn)
		return pn

	def testHSIC(self,x,y, ruido):
		power = 0
		n = len(x)
		for j in range(self.reps):
			aux = y + np.random.normal(0,ruido,len(x))
			aux = (aux -np.mean(aux))/np.std(aux)
			start_time = time.time()
			dhsic = dHSIC(kernelGausiano(x),kernelGausiano(aux))
			pval = self.computePvalue(dhsic)
			elapsed_time = time.time() - start_time
			self.times.append(elapsed_time)
			if(pval <= 0.01):
				power += 1./self.reps
				#print("Si",j,"              ",end ="\r")
		#print(statistic,var, end ="\r")
		#print(ruido,power,"                   ",end ="\r")
		return power

def resampling(n,b=100):
	x = np.random.normal(0,1, n)
	y = np.random.normal(0,1,n)
	valores = []
	for i in range(b):
		perm = permutation(n)
		valores.append(dHSIC(kernelGausiano(x),kernelGausiano(y[perm])))
	return valores
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
				norma[i,j] = np.exp(-(2*sigma)**(-2)* (X[i] - X[j])**2)
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

	return H.dot(K.dot(H))
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
	return np.trace(Kxc*Kyc)/np.power(np.shape(KX)[0]-1,2)

def null_samplesHsic(Kx,Ky,n):
	null_samples=np.zeros(n)
	for i in range(n):
		perm = permutation(np.shape(Ky)[0])
		Kpp = Ky[perm,:][:,perm]
		null_samples[0]=HSIC_U_statistic(Kx,Kpp)
	return null_samples

def HSIC_U_statistic_test(x,y,blocksize = 50, nblocks = 10):
	Btest = np.zeros(nblocks)
	n = len(x)
	for i in range(nblocks):
		indx1 = i*blocksize
		indx2 = indx1+blocksize
		kx = kernelGausiano(x[indx1:indx2])
		ky = kernelGausiano(y[indx1:indx2])
		Btest[i]= HSIC_U_statistic(kx,ky)
	Btest_Statistic = sum(Btest)/float(nblocks)
	kx = kernelGausiano(x)
	ky = kernelGausiano(y)
	Btest_nullVar = blocksize**2 * np.var(null_samplesHsic(kx,ky,nblocks))
	z_score = np.sqrt(n*nblocks)*Btest_Statistic/np.sqrt(Btest_nullVar)
	print("perm-pv",normaldist.sf(z_score))
	
	ft = HSIC_U_statistic(kx,ky)
	st =  HSIC_U_statistic(kx,kx)*  HSIC_U_statistic(ky,ky)
	r = ft/(np.sqrt(st))
	#test normaldist.sf(ustatistic) < alpha?
	return r
def Calcular_Esperanza_Var(Kx,Ky):
	# Btest = np.zeros(n)
	# for i in range(n):
	# 	perm = permutation(blocksize)
	# 	kx = kernelGausiano(x[perm])
	# 	ky = kernelGausiano(y[perm])
	# 	Btest[i] = HSIC_U_statistic(kx,ky)
	# return [np.mean(Btest),(np.std(Btest)*n**3./2)**2]
	m = np.shape(Kx)[0]
	np.fill_diagonal(Kx,1)
	np.fill_diagonal(Ky,1)
	#print(sum(sum(Kx)))
	mux = sum(sum(Kx))*(1./(m*(m-1)))
	muy = sum(sum(Ky))*(1./(m*(m-1)))
	mean = (1./m)*(1+ mux*muy - mux-muy)
	K = center_kernel_matrix(Kx,True)
	L = center_kernel_matrix(Ky,True)
	B = np.matrix(K.dot(L))
	B = np.power(B,-2)
	onesT = np.ones((1,m))
	ones = np.ones((m,1))
	var = onesT.dot((B-np.trace(B)).dot(ones))
	var = (2*(m-4)*(m-5))/(m*(m-1)*(m-2)*(m-3)) * var
	return [mean,var]

def gamma_valor(x,a,b):
	return np.power(x,a-1)*np.exp(-x/b)/(np.power(b,a)*factorial(np.floor(a)))
def dHSIC(Kx,Ky):
	m = np.shape(Kx)[0]
	term1 = np.ones((m,m))
	term2 = 1
	term3 = (2./m)*np.ones((1,m))
	term1 = term1 * Kx * Ky
	term2 = (1./(m**4))*np.sum(np.sum(Kx))*np.sum(np.sum(Ky))
	term3 = 1./(m**2)*term3*np.sum(Kx,0)*term3*np.sum(Ky,0)
	dhsic = (1./m**2)*np.sum(term1) + term2 + np.sum(term3)
	#print(dhsic)
	return dhsic

def resamplingTest(x,y,b = 100):
	pn = 1./(b+1)
	dhsic = dHSIC(kernelGausiano(x),kernelGausiano(y))
	count = 0.
	for i in range(b):
		perm = permutation(len(x))
		if (dhsic < dHSIC(kernelGausiano(x[perm]),kernelGausiano(y))):
			count +=1.0
	pn += count/(b+1)
	#print(pn)
	return pn

def testHSIC_0(x,y,ruido):
	power = 0
	reps = 10
	n = len(x)
	Kx = kernelGausiano(x)
	
	for j in range(reps):
		aux = y + np.random.normal(0,ruido,len(x))
		aux = (aux -np.mean(aux))/np.std(aux)
		Ky = kernelGausiano(aux)
		pval = gammaTest(dHSIC(Kx,Ky),Kx,Ky)
		#print("d",pval)
		hsic = HSIC_V_statistic(Kx,Ky)
		pval2 = gammaTest(hsic,Kx,Ky)
		#print("V",gammaTest(hsic,Kx,Ky))
		if(pval < 0.05 or pval2 < 0.05):
			power += 1./reps
			#print("Si",j,"              ",end ="\r")
	#print(statistic,var, end ="\r")
	#print(ruido,power,"                   ",end ="\r")
	return power
def gammaTest(hsic,Kx,Ky):
	e,var = Calcular_Esperanza_Var(Kx,Ky)

	print(e,var)
	a = e**2/var
	b = len(x)*var/e
	print(a,b)
	return 1-gamma.cdf(len(x)*hsic,a,b)
def circle(x):
  R = 0.5;
  x0 = 0.5;
  y0 = 0;
  factor = [np.power(-1,np.random.binomial(1,0.5)) for i in range(len(x))]
  aux = np.power((x-x0),2)
  y = y0 + np.power(np.power(R,2) - aux ,1./2)
  return y*factor
"""
n = 25

x = np.random.rand(n)
y = np.log(x)
x = (x - np.mean(x))/np.std(x)
#hsic = HSIC(n)


print("Res0\n",testHSIC_0(x,y,0))
print("Res1\n",testHSIC_0(x,y,1./9))
print("Res2\n",testHSIC_0(x,y,2./7))
print("Res3\n",testHSIC_0(x,y,3./5))

y = (x-0.5)**2
print("Res0\n",testHSIC_0(x,y,0))
print("Res1\n",testHSIC_0(x,y,1./9))
print("Res2\n",testHSIC_0(x,y,2./7))
print("Res3\n",testHSIC_0(x,y,3./5))
#y = (y - np.mean(y))/np.std(y)
#print(resamplingTest(x,y)) #Mayor que 0.05 son independientes
#r = HSIC_V_statistic(kernelGausiano(x),kernelGausiano(y))
#print("Vstatistic2",r)

#print(np.sqrt(np.log(6/0.95)/(0.24*n)))
y = np.copy(x) 
print("Res0\n",testHSIC_0(x,y,0))
print("Res1\n",testHSIC_0(x,y,1./9))
print("Res2\n",testHSIC_0(x,y,2./7))
print("Res3\n",testHSIC_0(x,y,3./5))
#Menor que 0.05 no son indeps
#estadistico = HSIC_U_statistic_test(x,y)
#r = HSIC_V_statistic(kernelGausiano(x),kernelGausiano(y))
#print(r)
x = np.random.normal(0,1, n)
y = np.random.normal(0,1,n)
print("Res0\n",testHSIC_0(x,y,0))
print("Res1\n",testHSIC_0(x,y,1./9))
print("Res2\n",testHSIC_0(x,y,2./7))
print("Res3\n",testHSIC_0(x,y,3./5))
"""