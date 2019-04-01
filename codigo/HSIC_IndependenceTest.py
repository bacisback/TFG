import numpy as np
from scipy.stats import gamma
from IndependenceTest import *

class HSIC_IndependenceTest(IndependenceTest):
	def __init__(sef,filas,columnas,titulos):
		super().__init__("HSIC",filas,columnas,titulos)
	def test(self,x,y,alpha):
		[testStat,thresh] = HSIC_test_gamma(x,y,alpha)
		if testStat > thresh:
			return 1
		else:
			return 0
	def generate_statistic(self,x,y):
		[testStat,thresh] = HSIC_test_gamma(x,y,0.05)
		return testStat


def HSIC_test_gamma(x,y,alpha):
	params = [-1,-1]
	m = x.size
	if x.ndim == 1:
		x = x.reshape((-1,1))
	if y.ndim == 1:
		y = y.reshape((-1,1))
	"""
	Sacamos los kernels de X e Y
	"""
	if params[0] == -1:
		size1 = len(x)
		if size1 > 100:
			xmed = x[0:100]
			size1 = 100
		else:
			xmed = x
		G = np.sum(xmed*xmed,1)
		Q = np.repeat(G,size1).reshape(size1,size1)
		R = Q.T
		dists = Q + R -2*np.dot(xmed,xmed.T)
		dists = dists - np.tril(dists)
		dists = dists.reshape(size1*size1,1)
		params[0] = np.sqrt(0.5*np.median(dists[dists>0]))
	if params[1] == -1:
		size1 = len(y)
		if size1 > 100:
			ymed = y[0:100]
			size1 = 100
		else:
			ymed = y
		G = np.sum(ymed*ymed,1)
		Q = np.repeat(G,size1).reshape(size1,size1)
		R = Q.T
		dists = Q + R -2*np.dot(ymed,ymed.T)
		dists = dists - np.tril(dists)
		dists = dists.reshape(size1*size1,1)
		params[1] = np.sqrt(0.5*np.median(dists[dists>0]))


	bone = np.ones((m,1))

	H = np.eye(m) -(1./m)*np.ones((m,m))

	K = rbf_dot(x,x,params[0])
	L = rbf_dot(y,y,params[1])

	Kc = (H.dot(K)).dot(H)
	Lc = (H.dot(L)).dot(H)

	testStat = 1./m * np.sum(Kc.T*Lc)

	varHsic = np.power((1./6 * Kc*Lc),2)
	varHsic = 1./m/(m-1) * (np.sum(varHsic) - np.sum(np.diag(varHsic)))
	varHsic = 72*(m-4)*(m-5)/m/(m-1)/(m-2)/(m-3) * varHsic

	K = K-np.diag(np.diag(K))
	L = L-np.diag(np.diag(L))
	
	
	
	muX = 1./m/(m-1)* bone.T.dot(K.dot(bone))
	muY = 1./m/(m-1)* bone.T.dot(L.dot(bone))


	mHsic = 1./m * ( 1+muX*muY -muX - muY)

	

	al = np.power(mHsic,2)*1./varHsic
	bet = varHsic*m*1./mHsic

	thresh = 1- gamma.cdf(1-alpha,al,bet)
	return [testStat,thresh]

"""
Funcion auxiliar para sacar el producto interno
"""
def rbf_dot(patterns1,patterns2,deg,Flag = True):
	size1 = len(patterns1)
	size2 = len(patterns2)

	G = np.sum(patterns1*patterns1,1)
	Q = np.repeat(G,size1).reshape(size1,size1)
	if Flag:
		R = Q.T
		H = Q+R-2*np.dot(patterns1,patterns2.T)
		H = np.exp(-H/2./(deg**2))
		return H

	H = patterns2*patterns2
	R = np.repeat(H.T,size1).reshape(size1,size1)
	H = Q+R-2*patterns1*patterns2.T
	H = np.exp(-H/2./(deg**2))
	return H

