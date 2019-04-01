import numpy as np
def Linear(x):
	return np.copy(x)
def Parabolic(x):
	return np.power((x-0.5),2)
def Cubic(x):
	y = np.power(2*(x-0.5),3) + np.power(2*(x-0.5),2) 
	return y
def Sin1(x):
	return np.sin(4*np.pi*x)
def Sin2(x):
	return np.sin(16*np.pi*x)
def root4(x):
	return np.power(x,0.25)
def circle(x):
	R = 0.5;
	x0 = 0.5;
	y0 = 0;
	factor = [np.power(-1,np.random.binomial(1,0.5)) for i in range(len(x))]
	aux = np.power((x-x0),2)
	y = y0 + np.power(np.power(R,2) - aux ,1./2)
	return y*factor
def aux_step(x):
	if x < 0.5:
		return 0.
	return 1.
def step(x):
	return map(aux_step,x)
def xsin(x):
	return x*np.sin(x)
def logarithm(x):
	return np.log(x)
def gausian(x):
	c = np.std(x)
	b = np.mean(x)
	return 1./(c*np.sqrt(2*np.pi))*np.exp(-np.power(x-b,2)/(2*np.power(c,2)))
def bivariate_gaussian(n,cov = [[1,0.5],[0.5,1]]):
	return np.random.multivariate_normal([0, 0], cov, n).T
def gaussian_multiply_uniform(n):
	x,y = np.random.multivariate_normal([0, 0], np.eye(2), n).T
	z = np.random.rand(n)
	return [np.dot(x,z), np.dot(y,z)]
def mixture_3_gaussians(n):
	z1 = np.random.multivariate_normal([0, 0], np.eye(2), n).T
	z2 = np.random.multivariate_normal([0, 0], [[1,0.8],[0.8,1]], n).T
	z3 = np.random.multivariate_normal([0, 0], [[1,-0.8],[-0.8,1]], n).T
	p = np.random.rand(n)

	z = (p<0.6).astype(float)*(z1-z2) + (p<0.8).astype(float)*z2 + (p>0.8).astype(float)*z3
	return z
def gaussian_multiplicative_noise(n):
	z = np.random.multivariate_normal([0, 0], np.eye(2), n).T
	e = np.random.multivariate_normal([0, 0], [[1,0.8],[0.8,1]], n).T
	return z*e

