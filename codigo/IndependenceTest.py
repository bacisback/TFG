from abc import abstractmethod, ABCMeta
import numpy as np
import matplotlib.pyplot as plt
import time

class IndependenceTest:
	__metaclass__ = ABCMeta

	def __init__(self,name,filas,columnas,titles = None):
		self.__name = name
		self.solutions = np.ones((filas,columnas))
		self.titles = titles
	def __str__(self):
		return self.__name
	def add_titles(self,titles):
		self.titles = titles
	def print(self,title = None):
		if title is None:
			title = self.__name
		else:
			title = title + self.__name
		text = './datos/'+title+'.txt'
		np.savetxt(text,self.solutions,delimiter="\t")
	def plot(self,ax=None,fila= None):
		if ax is None:
			ax = plt.gcf()
		if fila is None:
			ax.suptitle(self.__name)

			x = range(len(self.solutions[0,:]))
			for i in range(len(self.titles)):
				ax.add_subplot(4,3,i+1)
				plt.plot(x,self.solutions[i,:],label=str(self.titles[i]))
		else:
			ax.plot(range(0,len(self.solutions[fila,:])),self.solutions[fila,:])	
	@abstractmethod
	def test(self,x,y,alpha,statistic = False):
		"""
			Function which will perform the independence test.
			Args:
				x,y: one-dimensional arrays 
				with the given sets of data to perform the test.
				alpha: alpha of the test, 1-alpha = confidence 
			Return:
				1 if x and y are not idependent
				0 if x and y are independent
		"""

		pass
	@abstractmethod
	def generate_statistic(self,x,y):
		pass
	def test_tiempos(self,n,begin,end,perms = 100):
		self.times = np.zeros((3,n))
		division = np.linspace(begin,end,n).astype(int)
		mean = [0, 0]
		cov = [[1, 0], [0, 1]]  # diagonal covariance

		for i,d in enumerate(division):
			current = np.empty(perms)
			for j in range(perms):
				x,y = np.random.multivariate_normal(mean, cov, d).T
				start_time = time.time()
				self.test(x,y,0.05)
				current[j] = time.time() - start_time
			self.times[1,i] = np.mean(current)
			self.times[0,i] = np.percentile(current, 25)
			self.times[2,i] = np.percentile(current, 75)
		text = './datos/TIMES/'+self.__name+'tiempos.txt'
		np.savetxt(text,self.times,delimiter="\t")
		plt.plot(division,self.times[1,:],'r', label ="Mean")
		plt.plot(division,self.times[0,:],'b', label ="First Quantile")
		plt.plot(division,self.times[2,:],'b', label ="Third Quantile")
		plt.xlabel(Size)
		plt.ylabel(Time)
		plt.title(self.__name)
		plt.savefig("figuras/TIMES/"+self.__name+".png")

	def generate_histogram(self,n,size=500):
		self.dependant = np.empty(n)
		self.independant = np.empty(n)
		mean = [0, 0]
		cov = [[1, 0], [0, 1]]  # diagonal covariance
		for i in range(n):
			xd,yd = np.random.multivariate_normal(mean, cov, size).T
			xi,yi = np.random.multivariate_normal(mean, np.ones((2,2)), size).T
			self.dependant[i] = self.generate_statistic(xd,yd)
			self.independant[i] = self.generate_statistic(xi,yi)
		text = './datos/'+self.__name+'histograma.txt'
		DATA = np.stack((self.dependant,self.independant))
		np.savetxt(text,DATA,delimiter="\t")
		print(self.__name)
		
	def add_solution(self,row,column,solution):
		self.solutions[row,column] = solution

	def test_varing_size(self,generate_func,nin,nend,steps,row,perms=500):
		space = np.linspace(nin,nend,steps).astype(int) 
		for i in range(steps):
			
			sol = 0
			for _ in range(perms):
				[x,y] = generate_func(space[i])
				sol += self.test(x,y,0.05)
			sol = sol*1./perms
			self.solutions[row,i] = sol
	def test_rotation(self,anglein,angleend,steps,size,perms=500):
		angle = np.linspace(anglein,angleend,steps)
		for i in range(steps):
			sol = 0
			for _ in range(perms):
				x,y = rotation(size,angle[i]*np.pi)
				sol += self.test(x,y,0.05)
			sol = sol*1./perms
			self.solutions[1,i] = sol

