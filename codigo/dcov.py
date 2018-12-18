import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

def dcov(X,Y):
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
				A[j,i] = np.sqrt(np.power(X[i]-X[j],2))
				B[j,i] = np.sqrt(np.power(Y[i]-Y[j],2))
	
	
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
	dcovXX = dcovXX*1./n2
	dcovYY = dcovYY*1./n2
	V = dcovXX*dcovYY
	if V > np.finfo(float).eps:
		dcor = np.sqrt(dcov*1./np.sqrt(V))
	else:
		dcor = 0.0

	return [np.sqrt(dcov),n*dcov/S2,dcor,norm.ppf(1-0.05/2)**2]
	

n = 500
x = np.random.rand(n)

y = np.power(x,2)

print(dcov(x,y))
y = np.random.rand(n)
print(dcov(x,y))
#print(dcovS1_S2_S3(x,y))



