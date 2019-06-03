from IndependenceTest import *
from HSIC_IndependenceTest import *
from RDC_IndependenceTest import *
from DCOV_IndependenceTest import *
import numpy as np
import matplotlib.pyplot as plt
import threading
import concurrent.futures
from funciones import *
class Tester:
	def __init__(self,funciones,titulos,steps,n):
		self.funciones = funciones
		self.titulos = titulos
		self.l = len(self.funciones)
		self.n = n
		self.steps = steps
		self.tests = []
		#self.potencias= np.array((1,1,10,2,1,1,0.25,5,1,5,1,1))*0.5
	def add_test(self,test):
		self.tests.append(test)
	def plot(self):
		for test in self.tests:
			plt.figure()
			test.plot()
			plt.legend(loc='best')
		plt.show()
	def print(self,title=None):
		for test in self.tests:
			test.print(title)
	def simulate(self):
		threads = []
		for test in self.tests:
			test_thread_list = []
			for i in range(self.steps):
				#print(i*1./self.steps,end='\r')
				for j in range(self.l):
					thread = threading.Thread(
						target = self.simulate_dependant,
						args =(self.n,i*3./self.steps,j,i,test))
					thread.start()
					test_thread_list.append(thread)
				thread = threading.Thread(
						target = self.simulate_independant,
						args =(self.n,i*3./self.steps,self.l,i,test))
				thread.start()
				test_thread_list.append(thread)
				for thread in test_thread_list:
				
					thread.join()
			#threads.append(test_thread_list)
		#for list_of_threads in threads:
			
	def compute_times(self,n,begin,end):
		for test in self.tests:
			test.test_tiempos(n,begin,end)

	def generate_histograms(self,n,size=500):
		
		with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
			futures = {executor.submit(test.generate_histogram,n) for test in self.tests}
			concurrent.futures.wait(futures)
		

	def simulate_dependant(self,n,noise,row,column,test,perms =500):
		power = 0
		
		for _ in range(perms):
			
			x = np.random.rand(n)
			z = self.funciones[row](x)
			y = np.array([j for j in z]) + np.random.normal(0,noise,n)
			x = (x - np.mean(x))*1./np.std(x)
			y = (y - np.mean(y))*1./np.std(y)
			power += test.test(x,y,0.05)
			
		power = power * 1./perms
		
		test.add_solution(row,column,power)
	def simulate_independant(self,n,noise,row,column,test,perms =50):
		power = 0
		mean = [0, 0]
		cov = [[1, 0], [0, 1]]  # diagonal covariance
		for _ in range(perms):
			x, y = np.random.multivariate_normal(mean, cov, n).T
			y = y + np.random.normal(0,noise,n)
			x = (x - np.mean(x))*1./np.std(x)
			y = (y - np.mean(y))*1./np.std(y)
			power += test.test(x,y,0.05)
		power = power *  1./perms
		print(noise,end='\r')
		test.add_solution(row,column,power)

	def sample_size_test(self,ninit,nend,steps,generate_func):
		for i in range(len(generate_func)):
			with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
				futures = {executor.submit(test.test_varing_size,generate_func[i],ninit,nend,steps, i) for test in self.tests}
				concurrent.futures.wait(futures)
		
"""
functions = [bivariate_gaussian,gaussian_multiply_uniform,mixture_3_gaussians,gaussian_multiplicative_noise]
titles = ["bivariate gaussian","Gaussian multiply uniform", "Mixture of 3 gaussians","Gaussian multiplicative noise"]
steps = 7
ninit = 10
nend = 500
n = 5
tester = Tester(functions,titles,steps,n)
dcov = DCOV_IndependenceTest(len(titles),steps,titles)
rdc = RDC_IndependenceTest(len(titles),steps,titles)
hsic = HSIC_IndependenceTest(len(titles),steps,titles)
tester.add_test(dcov)
tester.add_test(rdc)
tester.add_test(hsic)
tester.sample_size_test(ninit,nend,steps,functions)
tester.plot()
tester.print("testPerms/Asymptotic/")
"""
titles = ["Linear","Parabolic","Cubic","Sin1","Sin2","root4","circle","step","xsin","logarithm","gausian","2D Gaussian"]
steps = 10
n = 200
tester = Tester([Linear,Parabolic,Cubic,Sin1,Sin2,root4,circle,step,xsin,logarithm,gausian],titles,steps,n)
dcov = DCOV_IndependenceTest(len(titles),steps,titles)
#dcov.test_varing_size(bivariate_gaussian,ninit,nend,steps,1)
rdc = RDC_IndependenceTest(len(titles),steps,titles)
hsic = HSIC_IndependenceTest(len(titles),steps,titles)
tester.add_test(dcov)
tester.add_test(rdc)
tester.add_test(hsic)
tester.simulate()
tester.plot()
tester.print("exp1Asint/2")