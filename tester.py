from testRDC import testRDC
import numpy as np
import matplotlib.pyplot as plt

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
#cancor(norm(40).reshape(4,10),norm(50).reshape(5,10))
n = 500
#np.random.seed(1)
x = np.random.rand(n)

funciones = [Linear,Parabolic,Cubic,Sin1,Sin2,root4,circle,step,xsin,logarithm,gausian]
titulos = ["lineal","Parabolic","Quadratic","Sin(4pix)","Sin(16pix)","fourth root","circle","step","xsin(x)","logarithm","gausian","2D gausian"]
shape = [10,50]
l = len(funciones)
y = np.zeros((l+1,n))
a = [1,1,10,2,1,1,0.25,5]
solutions = np.zeros((l+1,30))
solutionsHsic = np.zeros((l+1,30))
corrs = np.zeros((l+1,30))
for i in range(l):
    z = funciones[i](x)
    y[i]=np.array([j for j in z])
    #solutions[1,i] = Ahsic(x,y[i])
  
auxX = (x - np.mean(x))/np.std(x)
"""for i in range(l):
    ax = plt.subplot(4,3,i+1)
    ax.set_title(str(titulos[i]))
    aux = (y[i] - np.mean(y[i]))/np.std(y[i])
    plt.plot(auxX,aux,'o',markersize=0.5)
"""
x = auxX
auxX  = np.random.normal(0,1, n)
y[l] = np.random.normal(0,1, n)

#ax = plt.subplot(4,3,12)
#ax.set_title(str(titulos[l]))
#plt.plot(auxX,y[l,:],'o',markersize=0.5)
#plt.show()

for i in range(0,30):
    for j in range(l+1):
        aux = y[j] + np.random.normal(0,(i**2)*1./2700,n)
        aux = (aux -np.mean(aux))/np.std(aux)
        #path, beta, A, lam = hsiclasso(x, aux, numFeat=5,ykernel='Delta')
        if j == l:
            solutions[j,i] = testRDC(auxX,aux)
            #solutionsHsic[j,i] = HSIC_U_statistic_test(auxX,aux)
            #print(solutionsHsic[j,i])
            corrs[j,i] = np.abs(np.corrcoef(auxX,aux)[0,1])
        else:
            solutions[j,i] = testRDC(x,aux)
            #solutionsHsic[j,i] = HSIC_U_statistic_test(x,aux)

            corrs[j,i] = np.abs(np.corrcoef(x,aux)[0,1])
for i in range(len(y)):
    ax = plt.subplot(4,3,i+1)
    plt.ylim((0,1.1))
    ax.set_title(str(titulos[i]))
    plt.plot(range(0,30),solutions[i,:],label="RDC")
    plt.plot(range(0,30),corrs[i,:],label="corr")
    #plt.plot(range(0,30),solutionsHsic[i,:],label="HSIC")
plt.legend(loc='best')
plt.show()