import numpy as np
from IndependenceTest import *
from scipy.stats import norm
from numpy.random import permutation

class DCOV_IndependenceTest(IndependenceTest):
	def __init__(sef,filas,columnas,titulos):
		super().__init__("DCOV",filas,columnas,titulos)
	def test(self,x,y,alpha):
		[DCOV,DCOR,statistic,thresh] = dcov(x,y,alpha)
		if statistic > thresh:
			return 1
		else:
			return 0
	def generate_statistic(self,x,y):
		[DCOV,DCOR,statistic,thresh] = dcov(x,y,0.05)
		return DCOV

def dcov(X,Y,alpha,R = None):
	n = len(X)
	n2 = n*n
	A = np.zeros((n,n))
	B = np.zeros((n,n))
	a_filas = 0
	b_filas = 0
	a = 0
	b = 0
	dcov = 0
	dcovXX = 0
	dcovYY = 0
	for i in range(n):
		for j in range(i):
			if i != j:
				A[i,j] = np.sqrt(np.power(X[i]-X[j],2))
				B[i,j] = np.sqrt(np.power(Y[i]-Y[j],2))
				A[j,i] = A[i,j]
				B[j,i] = B[i,j]
	
	
	a_columnas = np.zeros(n)
	b_columnas = np.zeros(n)
	for i in range(n):
		a_columnas[i] = np.sum(A[i,:])
		b_columnas[i] = np.sum(B[i,:])
	a = np.sum(a_columnas)*1./n2
	b = np.sum(a_columnas)*1./n2
	a_columnas = a_columnas*1./n
	b_columnas = b_columnas*1./n
	for i in range(n):
		for j in range(n):
			A[i,j] = A[i,j] - a_columnas[i] - a_columnas[j] + a
			B[i,j] = B[i,j] - b_columnas[i] - b_columnas[j] + b
			dcov += A[i,j]*B[i,j]
			dcovXX += A[i,j]*A[i,j]
			dcovYY += B[i,j]*B[i,j]
	S2 = a*b
	dcov = dcov*1./n2
	dcov = np.sqrt(dcov)

	dcovXX = dcovXX*1./n2
	dcovXX = np.sqrt(dcovXX)

	dcovYY = dcovYY*1./n2
	dcovYY = np.sqrt(dcovYY)

	V = dcovXX*dcovYY

	if V > np.finfo(float).eps:
		dcor = np.sqrt(dcov*1./np.sqrt(V))
	else:
		dcor = 0.0
	"""
	R param let us choose between permutation or asimpthotic 
	method for calculating the pvalue
	"""
	if R is None:
		return [dcov,dcor,n*dcov*dcov/S2,norm.ppf(1-alpha/2)**2]
	else:
		pvalue = 0.0
		for i in range(R):
			dcov_aux = 0.0
			perm = permutation(n)
			for k in range(n):
				K = perm[k]
				for j in range(n):
					J = perm[j]
					dcov_aux += A[k][j]*B[K][J]
			dcov_aux = dcov_aux* 1./n2
			dcov_aux = np.sqrt(dcov_aux)
			if(dcov_aux >= dcov):
				pvalue += 1
		pvalue = pvalue*1./R
		return [dcov,dcor,pvalue]