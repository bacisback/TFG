t = BasedOnGauss_IndependenceTest(200,100,[Linear,Parabolic,Cubic,Sin1,Sin2,root4,circle,step,xsin,logarithm,gausian],np.linspace(0,3,10),100)
t.test()
t.print("datos/permutaciones200_1")