from IndependenceTestTester import *
from IndependenceTest import *
from HSIC_IndependenceTest import *
from RDC_IndependenceTest import *
from DCOV_IndependenceTest import *
import numpy as np
import matplotlib.pyplot as plt
import threading
import concurrent.futures
from funciones import *
n = 500
functions = [Linear,Parabolic,Cubic,Sin1,Sin2,root4,circle,step,xsin,logarithm,gausian]
titles = ["lineal","Parabolic","Quadratic","Sin(4pix)","Sin(16pix)","fourth root","circle","step","xsin(x)","logarithm","gausian","2D gausian"]
steps = 30

tester = Tester(functions,titles,steps,n)
dcov = DCOV_IndependenceTest(len(titles),steps,titles)
#rdc = RDC_IndependenceTest(len(titles),steps,titles)
#hsic = HSIC_IndependenceTest(len(titles),steps,titles)
tester.add_test(dcov)
#tester.add_test(rdc)
#tester.add_test(hsic)
tester.generate_histograms(500)