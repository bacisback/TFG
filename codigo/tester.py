
from hsicGretton import testHSIC_0, hsicTestGamma
from testRDC import testRDC1Var, testRDC,RDCtester
import numpy as np
import matplotlib.pyplot as plt
import threading

class tester:
    def __init__(self,n,hsic = False,rdc=False, cors = False, Numero = False):
        self.funciones = [Linear,Parabolic,Cubic,Sin1,Sin2,root4,circle,step,xsin,logarithm,gausian]
        self.titulos = ["lineal","Parabolic","Quadratic","Sin(4pix)","Sin(16pix)","fourth root","circle","step","xsin(x)","logarithm","gausian","2D gausian"]
        self.l = len(self.funciones)
        self.n = n
        if Numero:
           
            self.x = x =  np.random.rand(n)
            self.y = np.zeros((self.l+1,n))
            for i in range(self.l):
                z = self.funciones[i](x)
                self.y[i]=np.array([j for j in z])
            self.auxX  = np.random.normal(0,1, n)
            self.y[self.l] = np.random.normal(0,1, n)
        self.hsicT = hsic
        self.rdcT = rdc
        if hsic:
            self.solutionsHsic = np.ones((self.l+1,30))
        self.corrs = np.zeros((self.l+1,30))
        self.potencias= np.array((0.5,0.25,1,1,0.001,0.5,0.25,1,0.75,1,0.75,1))*0.5
        if rdc:
            self.solutions = np.ones((self.l+1,30))
            #self.rdcTester = RDCtester(n)
            #self.solutionsRDCperm = np.zeros((self.l+1,30))
        """
        self.rdcTester = RDCtester(n)
        if hsic:
            self.hsic = HSIC(n)
        """ 
       
        

    def plot(self):
        for i in range(len(self.titulos)):
            ax = plt.subplot(4,3,i+1)
            plt.ylim((-0.1,1.1))
            ax.set_title(str(self.titulos[i]))
            if self.rdcT:
                plt.plot(range(0,30),self.solutions[i,:],label="RDC")
                #plt.plot(range(0,30),self.solutionsRDCperm[i,:],label="RDC permutation")
            if self.hsicT:
                plt.plot(range(0,30),self.solutionsHsic[i,:],label="HSIC")
            #plt.plot(range(0,30),self.corrs[i,:],label="corr")
            #plt.plot(range(0,30),solutionsHsic[i,:],label="HSIC")
        plt.legend(loc='best')
        #plt.savefig('testplot.png')
        plt.show()

    def print(self):
        #np.savetxt('dataHsic',self.solutionsHsic,delimiter="\t")
        np.savetxt('dataRDC',self.solutions,delimiter="\t")
    def sim(self):
        #Hthreads = []
        RDCthreads = []
        rdcPerms = []
        for p in range(1,4):
            for i in range((p-1)*10,p*10):
                for j in range(self.l+1):
                    
                    
                    if j == self.l:
                        if self.hsicT:
                            h = threading.Thread(target = self.HsicIndependientes, args = (self.n,i*1./10*self.potencias[j] ,j,i))
                            h.start()
                            Hthreads.append(h)
                        if self.rdcT:
                            r = threading.Thread(target = self.RDCIndependientes, args = (self.n,i*1./10 ,j,i))
                        #rp = threading.Thread(target = self.sim_RDCperm, args = (self.auxX,self.y[j],i*1./10*self.potencias[j] ,j,i))
                        
                        
                            r.start()
                        #rp.start()
                        #rdcPerms.append(rp)
                        
                            RDCthreads.append(r)
                    else:
                        if self.hsicT:
                            h = threading.Thread(target = self.HsicConNumero, args = (self.n,i*1./10*self.potencias[j] ,j,i))
                            h.start() 
                            Hthreads.append(h)
                        if self.rdcT:
                            r = threading.Thread(target = self.RDCconNumero, args = (self.n,i*1./10,j,i))
                            #rp = threading.Thread(target = self.sim_RDCperm, args = (self.x,self.y[j],i*1./10*self.potencias[j],j,i))
                           
                            r.start()
                            #rp.start()
                           
                            RDCthreads.append(r)
                            #rdcPerms.append(rp)
            if self.hsicT:
                for h in Hthreads:
                    h.join()
        if self.rdcT:
            for r in RDCthreads:
                r.join()
        """
        for rp in rdcPerms:
            rp.join()
        print("TIEMPOS:\n")
        print("HSIC",end ="\t")
        print(np.mean(self.hsic.times))
        print("RDC",end ="\t")
        print(np.mean(self.times))
        print("RDCperms",end ="\t")
        print(np.mean(self.rdcTester.times))
        """
    def HsicConNumero(self,n,ruido,j,i):
        err = 0
        for _ in range(100):
            x = np.random.rand(n)
            z = self.funciones[j](x)
            y = np.array([v for v in z]) + np.random.normal(0,ruido,n)
            x = (x - np.mean(x))*1./np.std(x)
            y = (y - np.mean(y))*1./np.std(y)
            [stat,pvalue,thresh] = hsicTestGamma(x,y,0.05)
            if pvalue < 0.05:
                err += 1
        print(ruido , end="\r")
        self.solutionsHsic[j,i] = err*0.01
    def HsicIndependientes(self,n,ruido,j,i):
        err = 0
        for _ in range(100):
            x = np.random.normal(0,1, n)
            y = np.random.normal(0,1, n) + np.random.normal(0,ruido,n)
            x = (x - np.mean(x))*1./np.std(x)
            y = (y - np.mean(y))*1./np.std(y)
            [stat,pvalue,thresh] = hsicTestGamma(x,y,0.05)
            if pvalue < 0.05:
                err += 1
        print(ruido , end="\r")
        self.solutionsHsic[j,i] = err*0.01
    def RDCconNumero(self,n,ruido,j,i):
        err = 0
        for _ in range(100):
            x = np.random.rand(n)
            z = self.funciones[j](x)
            y = np.array([v for v in z]) + np.random.normal(0,ruido,n)
            x = (x - np.mean(x))*1./np.std(x)
            y = (y - np.mean(y))*1./np.std(y)
            p = testRDC1Var(x,y)
            if p < 0.05:
                err +=1
        print(ruido , end="\r")
        self.solutions[j,i] = err*0.01
    def RDCIndependientes(self,n,ruido,j,i):
        err = 0
        for _ in range(100):
            x = np.random.normal(0,1, n)
            y = np.random.normal(0,1, n) + np.random.normal(0,ruido,n)
            x = (x - np.mean(x))*1./np.std(x)
            y = (y - np.mean(y))*1./np.std(y)
            p = testRDC1Var(x,y)
            if p < 0.05:
                err +=1
        print(ruido , end="\r")
        self.solutions[j,i] = err*0.01

    def sim_HSIC(self,x,y,ruido,j,i):
        self.solutionsHsic[j,i] = testHSIC_0(x,y,ruido,i,j )
    def sim_RDC(self,x,y,ruido,j,i):

        self.solutions[j,i],times = testRDC(x,y,ruido)
        self.times.extend(times)
    def sim_RDCperm(self,x,y,ruido,j,i):
        self.solutionsRDCperm[j,i] = self.rdcTester.TestRDC(x,y,ruido)
        



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




hsic = False
rdc = True
cors = False
Numero = False
t = tester(500,hsic,rdc,cors,Numero)
t.sim()
t.plot()
t.print()
"""
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
  
x= (x - np.mean(x))/np.std(x)

hsic = HSIC(x)
for i in range(l):
    ax = plt.subplot(4,3,i+1)
    ax.set_title(str(titulos[i]))
    aux = (y[i] - np.mean(y[i]))/np.std(y[i])
    plt.plot(auxX,aux,'o',markersize=0.5)

auxX  = np.random.normal(0,1, n)
y[l] = np.random.normal(0,1, n)

#ax = plt.subplot(4,3,12)
#ax.set_title(str(titulos[l]))
#plt.plot(auxX,y[l,:],'o',markersize=0.5)
#plt.show()

for i in range(0,30):
    for j in range(l+1):
        aux = y[j] + np.random.normal(0,i/10,n)
        aux = (aux -np.mean(aux))/np.std(aux)
        #path, beta, A, lam = hsiclasso(x, aux, numFeat=5,ykernel='Delta')
        if j == l:
            solutions[j,i] = hsic.testHSIC(auxX,y[j],i/10 )
            #solutionsHsic[j,i] = HSIC_U_statistic_test(auxX,aux)
            #print(solutionsHsic[j,i])
            corrs[j,i] = np.abs(np.corrcoef(auxX,aux)[0,1])
        else:
            solutions[j,i] = hsic.testHSIC(x,y[j],i/10 )
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
plt.savefig('testplot.png')
f=open("hsic.txt","w")
f.write(solutions)
f.close()
"""