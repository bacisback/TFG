import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm 
from scipy.optimize import curve_fit
"""
Funcion de HSIC implementada en python siguiendo el cofigo de Gretton
Suponemos que los arrays siempre van a ser unidimensionales.

De lo contrario R = np.repeat(G.T,size1).reshape(size1,size1)
"""
def hsicTestGamma(x,y,alpha,params = [-1,-1]):
	m = len(x)
	x = x.reshape((-1,1))
	y = y.reshape((-1,1))
	if params[0] == -1:
		size1 = len(x)
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
		size1 = len(y)
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

	p_value = 1-gamma.cdf(testStat,al,bet)
	thresh = 1- gamma.cdf(1-alpha,al,bet)
	return [testStat,p_value,thresh]

"""
Funcion auxiliar para sacar el producto interno"""
def rbf_dot(patterns1,patterns2,deg,Flag = True):
	size1 = len(patterns1)
	size2 = len(patterns2)

	G = patterns1*patterns1
	Q = np.repeat(G,size1).reshape(size1,size1)
	if Flag:
		R = Q.T
		H = Q+R-2*patterns1*patterns2.T
		H = np.exp(-H/2./(deg**2))
		return H

	H = patterns2*patterns2
	R = np.repeat(H.T,size1).reshape(size1,size1)
	H = Q+R-2*patterns1*patterns2.T
	H = np.exp(-H/2./(deg**2))
	return H
def testHSIC_0(x,y,ruido,k=None,j=None):
	x = (x - np.mean(x))/np.std(x)
	#print(ruido)
	err=np.zeros(50)
	times = np.zeros(50)
	for i in range(50):
		aux = y + np.random.normal(0,ruido,len(x))
		aux = (aux -np.mean(aux))/np.std(aux)
		start_time = time.time()
		[stat,pvalue,thresh] = hsicTestGamma(x,aux,0.05)
		elapsed_time = time.time() - start_time
		err[i] = pvalue < 0.05
		times[i] = elapsed_time
	print(k , j , "          ",end = "\r")
	return np.mean(err)

def testTimes(n):
	x = np.random.normal(0,1,n)
	y = np.random.normal(0,1,n)
	times = np.zeros(100)
	for i in range(100):
		start_time = time.time()
		hsicTestGamma(x,y,0.05)
		elapsed_time = time.time() - start_time
		times[i] = elapsed_time
	return times

def circle(x):
	R = 0.5;
	x0 = 0.5;
	y0 = 0;
	factor = np.array([np.power(-1,np.random.binomial(1,0.5)) for i in range(len(x))])
	aux = np.power((x-x0),2)
	y = y0 + np.power(np.power(R,2) - aux ,1./2)
	return y*factor

def func(x,a,b,c):
	return a*x**2+b*x+c
"""
n = 100
t = np.linspace(50,1000,n).astype(int)
mean = np.zeros(n)
IC0 = np.zeros(n)
IC1 = np.zeros(n)
for i,x in  enumerate(t):
	print(x,end="\r")
	times = testTimes(x)
	mean[i] = np.mean(times)
	IC0[i] = np.percentile(times,5)
	IC1[i] = np.percentile(times,95)

lowess = sm.nonparametric.lowess(mean, t, frac=0.1) 
plt.figure()
plt.plot(t,mean,label="media",color="k")
plt.plot(t,IC0,'-',label="0.05%",color="r")
plt.plot(t,IC1,'-',label="0.95%",color="r")
plt.plot(lowess[:,0],lowess[:,1],'-',color = 'c',label="Media suavizada")
plt.legend(loc='best')
plt.figure()
popt, pcov = curve_fit(func, lowess[:,0],lowess[:,1])
plt.plot(lowess[:,0], func(lowess[:,0], *popt), 'g--',label='fit suavizada: a=%f, b=%f, c=%f'%tuple(popt))
plt.plot(lowess[:,0],lowess[:,1],'-',color = 'c',label="Media suavizada")
popt, pcov = curve_fit(func, t,mean)
plt.plot(t,func(t,*popt),'r-',label = 'fit sin suavizar: a=%f, b=%f, c=%f'%tuple(popt))
plt.legend(loc='best')
plt.show()
"""
"""
n = 500
x = np.random.rand(n)

y = np.log(x)


print("Res0\n",testHSIC_0(x,y,5./4))
print("Res1\n",testHSIC_0(x,y,6./4))
print("Res2\n",testHSIC_0(x,y,3./2))
print("Res3\n",testHSIC_0(x,y,9./5))

y = (x-0.5)**2
print("Res0\n",testHSIC_0(x,y,5./4))
print("Res1\n",testHSIC_0(x,y,6./4))
print("Res2\n",testHSIC_0(x,y,3./2))
print("Res3\n",testHSIC_0(x,y,9./5))

y = circle(x)
#print(y)
#plt.plot(x,y,'o')
#plt.show()
print("Res0\n",testHSIC_0(x,y,0))
print("Res1\n",testHSIC_0(x,y,2./4))
print("Res2\n",testHSIC_0(x,y,2./1.5))
print("Res3\n",testHSIC_0(x,y,4./5))
"""