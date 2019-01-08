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
