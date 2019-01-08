from IndependenceTest import *
from HSIC_IndependenceTest import *
from RDC_IndependenceTest import *
from DCOV_IndependenceTest import *
import numpy as np
import matplotlib.pyplot as plt
import threading
from funciones import *
class Tester:
	def __init__(self,funciones,titulos,steps,n):
		self.funciones = funciones
		self.titulos = titulos
		self.l = len(self.funciones)
		self.n = n
		self.steps = steps
		self.tests = []
		self.potencias= np.array((1,1,10,2,1,1,0.25,5,1,5,1,1))*0.5
	def add_test(self,test):
		self.tests.append(test)
	def plot(self):
		for test in self.tests:
			plt.figure()
			test.plot()
			plt.legend(loc='best')
		plt.show()
	def print(self):
		for test in self.tests:
			test.print()
	def simulate(self):
		threads = []
		for test in self.tests:
			test_thread_list = []
			for i in range(self.steps):
				#print(i*1./self.steps,end='\r')
				for j in range(self.l):
					thread = threading.Thread(
						target = self.simulate_dependant,
						args =(self.n,i*3./self.steps*self.potencias[j],j,i,test))
					thread.start()
					test_thread_list.append(thread)
				thread = threading.Thread(
						target = self.simulate_independant,
						args =(self.n,i*3./self.steps*self.potencias[j],self.l,i,test))
				thread.start()
				test_thread_list.append(thread)
				for thread in test_thread_list:
				
					thread.join()
			#threads.append(test_thread_list)
		#for list_of_threads in threads:
			

	def simulate_dependant(self,n,noise,row,column,test):
		power = 0
		n = 100
		for _ in range(n):
			
			x = np.random.rand(n)
			z = self.funciones[row](x)
			y = np.array([j for j in z]) + np.random.normal(0,noise,n)
			x = (x - np.mean(x))*1./np.std(x)
			y = (y - np.mean(y))*1./np.std(y)
			power += test.test(x,y,0.05)
			
		power = power * 1./n
		
		test.add_solution(row,column,power)
	def simulate_independant(self,n,noise,row,column,test):
		power = 0
		mean = [0, 0]
		cov = [[1, 0], [0, 1]]  # diagonal covariance
		n = 100
		for _ in range(n):
			x, y = np.random.multivariate_normal(mean, cov, n).T
			y = y + np.random.normal(0,noise,n)
			x = (x - np.mean(x))*1./np.std(x)
			y = (y - np.mean(y))*1./np.std(y)
			power += test.test(x,y,0.05)
		power = power *  1./n
		print(noise,end='\r')
		test.add_solution(row,column,power)

n = 500
functions = [Linear,Parabolic,Cubic,Sin1,Sin2,root4,circle,step,xsin,logarithm,gausian]
titles = ["lineal","Parabolic","Quadratic","Sin(4pix)","Sin(16pix)","fourth root","circle","step","xsin(x)","logarithm","gausian","2D gausian"]
steps = 30
tester = Tester(functions,titles,steps,n)
dcov = DCOV_IndependenceTest(len(titles),steps,titles)
rdc = RDC_IndependenceTest(len(titles),steps,titles)
hsic = HSIC_IndependenceTest(len(titles),steps,titles)
tester.add_test(dcov)
tester.add_test(rdc)
tester.add_test(hsic)
tester.simulate()
tester.plot()
tester.print()
