from abc import abstractmethod, ABCMeta
import numpy as np
import matplotlib.pyplot as plt

class IndependenceTest:
	__metaclass__ = ABCMeta

	def __init__(self,name,filas,columnas,titles = None):
		self.__name = name
		self.solutions = np.ones((filas,columnas))
		self.titles = titles
	def add_titles(self,titles):
		self.titles = titles
	def print(self):
		text = './datos/'+self.__name+'.txt'
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
	def test(self,x,y,alpha):
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
	def test_tiempos(self,n):
		pass
	def add_solution(self,row,column,solution):
		self.solutions[row,column] = solution