#!/usr/bin/python2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from scipy.stats import rankdata
from pyHSICLasso import HSICLasso

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1,len(x)+1)/float(len(x))
    y  = np.ones(len(x))
    xs = xs.tolist()
    for i in range(len(ys)):
        y[i] = ys[xs.index(x[i])]
    return y
def tau(x):
    return 1/(1+ np.exp(-x))
def norm(k,s=1):
    ret =  np.random.normal(0,s,k)
    return ret
def rdc(x, y, f=np.sin, k=20, s=1/6., n=5):
    
    if n>1:
        v = []
        for i in range(n):
            v.append(rdc(x,y,f,k,s,1))
        return np.median(v)

    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued

    k0 = k
    lb = 1
    ub = k
    while True:
        k = int(k)
        #print(k)

        # Compute canonical correlations
        Cxx = C[:k,:k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.inv(Cxx), Cxy),
                                        np.dot(np.linalg.inv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) / 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) / 2

    return np.sqrt(np.max(eigs))

def rdc1(x,y,k=10,s=0.2):
    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)
    X = np.sin()
    """print rcancor(np.sin(X),np.sin(Y))
        return 0"
        """
    cca = CCA(n_components=1)
    xc,yc = cca.fit_transform(X,Y)
    result = np.corrcoef(xc.T,yc.T)[0,1]
    print(result)
def rdc2(x,y,k=10,s=0.2,n=5):
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc2(x, y, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)
    lx = len(x)
    ly = len(y)
    x = np.concatenate((ecdf(x).reshape(-1,1),
        np.ones(lx).reshape(-1,1)),axis=1)
    y = np.concatenate((ecdf(y).reshape(-1,1),
        np.ones(ly).reshape(-1,1)),axis=1)
    
    nx = x.shape[1]
    ny = y.shape[1]
    wx = norm(nx*k,s).reshape(nx,k)
    wy = norm(ny*k,s).reshape(ny,k)
    wxs = np.matmul(x,wx)
    wys = np.matmul(y,wy)
    #fX = np.sin(wxs)
    #fY = np.sin(wys)
    #res = cancorCopy(fX,fY,k)
    #return res
    wxs = np.concatenate((np.cos(wxs),np.sin(wxs)),axis=1)
    wys = np.concatenate((np.cos(wys),np.sin(wys)),axis=1)
    """d = rcancor(wxs,wys)
    print("mio",d)"""
    cca = CCA(n_components=1)
    cca.fit(wxs,wys)

    xc,yc = cca.transform(wxs,wys)
    #result1 = cca.score(xc,yc)
    result2 = np.corrcoef(xc.T,yc.T)[0,1]
    #print(result1,result2)
    return result2
def cancorCopy(x,y,k):
    C = np.cov(np.hstack([x, y]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued

    k0 = k
    lb = 1
    ub = k
    while True:
        k = int(k)
        #print(k)

        # Compute canonical correlations
        Cxx = C[:k,:k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.inv(Cxx), Cxy),
                                        np.dot(np.linalg.inv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) / 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) / 2

    return np.sqrt(np.max(eigs))

def cancor(x,y):
    q = len(x)
    p = len(y)

    #Singular value decomposition para sacar K = xx(1/2)xyyy(1/2)
    xy= np.cov(x,y)
    """print("Matriz de covarianzas\n")
    s = [[str(e) for e in row] for row in xy]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print '\n'.join(table)
    print("\nxx\n")"""
    xx = xy[:q,:q]
    #print(xx)
    #print("\nyy\n")
    yy = xy[q+1:,q+1:]
    #print(yy)
    #print("\nxy\n")
    exy = xy[:q,q+1:]
    eyx = xy[q+1:,:q]
    #print(exy)
    #print("\nInversas:\n")

    ixx = np.linalg.inv(np.sqrt(xx))
    iyy = np.linalg.inv(np.sqrt(yy))
    #print("\nK\n")
    K = np.dot(np.dot(ixx,exy),iyy)
    
    #print(K)
    t,d,delta = np.linalg.svd(K)
    #print("\ngamma\n")
    #print(t)
    #print("\ndelta\n")
    #print(delta)
    """n = len(t) 
    a = np.zeros((n,q))
    #print("\n")
    #print("\neigenvalues\n")
    #print(np.sqrt(d))
#   print("\neigenvectors\n")
    #Calculo de los vectores de correlacion
    for i in range(n):
        a[i] = ixx.dot(t[:,i])
    #print(a)
    n = len(delta) 
    b = np.zeros((n,p))
    for i in range(n):
        b[i] = iyy.dot(delta[:,i])
    #print(b)"""
    return d[0]
def rcancor(x,y, xcenter = True, ycenter = True):
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    nr = np.size(x,0)
    if (nr != np.size(y,0)):
        raise NameError('number of rows of x and y are not equal')
    ncx = np.size(x,1)
    ncy = np.size(y,1)
    if(nr==0 or ncx==0 or ncy==0):
        raise NameError('dimension 0 in x or y')
    if(xcenter):
        xcenter = x.mean(axis=0)
        x = x - np.tile(xcenter,(nr,1))
    else:
        xcenter = np.repeat(0,ncx)
    if(ycenter):
        ycenter = y.mean(axis=0)
        y = y - np.tile(ycenter,(nr,1))
    else:
        ycenter = np.repeat(0,ncy)
    qx,rx = np.linalg.qr(x)

    qy,ry = np.linalg.qr(y)
    dx = np.linalg.matrix_rank(qx)
    dy = np.linalg.matrix_rank(qy)
    if (dx==0 or dy==0):
        raise NameError('rank of x or y is 0')
    aux = np.eye(nr,dy)
    
    K = qx.T*qy
    l = K.shape
    t,d,delta = np.linalg.svd(K[:dx,:])
    return d[0]
def Ahsic (x,y,reg = 5):
    if type(x) is not np.ndarray:
        print ("error x")
        return 
    if type(y) is not np.ndarray:
        print ("error y")
        return
    hsic_lasso = HSICLasso()
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    z = np.column_stack([x, y]).reshape(2,500)
    #print(z)
    hsic_lasso.input(z,np.array([0,1]))
    hsic_lasso.regression(reg)
    #print(hsic_lasso.get_index())
    hsic_lasso.classification(10)
    #hsic_lasso.plot()
    print  (hsic_lasso.get_index_score())


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

l = len(funciones)
y = np.zeros((l+1,n))
a = [1,1,10,2,1,1,0.25,5]
solutions = np.zeros((l+1,30))
corrs = np.zeros((l+1,30))
for i in range(l):
    z = funciones[i](x)
    y[i]=np.array([j for j in z])
    #solutions[1,i] = Ahsic(x,y[i])
    
auxX = (x - np.mean(x))/np.std(x)
for i in range(l):
    ax = plt.subplot(4,3,i+1)
    ax.set_title(str(titulos[i]))
    aux = (y[i] - np.mean(y[i]))/np.std(y[i])
    plt.plot(auxX,aux,'o',markersize=0.5)
    Ahsic(auxX,aux)
x = auxX
auxX  = np.random.normal(0,1, n)
y[l] = np.random.normal(0,1, n)

ax = plt.subplot(4,3,12)
ax.set_title(str(titulos[l]))
plt.plot(auxX,y[l,:],'o',markersize=0.5)
plt.show()

for i in range(0,30):
    for j in range(l+1):
        aux = y[j] + np.random.normal(0,(i**2)/900,n)
        aux = (aux -np.mean(aux))/np.std(aux)
        #path, beta, A, lam = hsiclasso(x, aux, numFeat=5,ykernel='Delta')
        if j == l:
            solutions[j,i] = rdc(auxX,aux)
            corrs[j,i] = np.corrcoef(auxX,aux)[0,1]
        else:
            solutions[j,i] = rdc2(x,aux)
            corrs[j,i] = np.abs(np.corrcoef(x,aux)[0,1])
for i in range(len(y)):
    ax = plt.subplot(4,3,i+1)
    ax.set_title(str(titulos[i]))
    plt.plot(range(0,30),solutions[i,:])
    plt.plot(range(0,30),corrs[i,:])
plt.show()
#x=np.matrix([[1.41,-1.11],[-1.11,1.19]])
#y=np.matrix([[0.78,-0.71,-0.9,-1.04,-0.95,0.18],[-0.42,0.82,0.77,0.9,1.12,0.11]])
#Ix,Iy=rdc(x,y,5)
#plt.scatter(Ix,Iy)
#print(Ix,Iy)
#x = datos[:4].reshape(2,2)
#y = datos[4:40].reshape(6,6)
#xy = np.cov(x)

# sigma = 1
# mu = 0
# s = np.random.normal(0, 1, 1000)
# ount, bins, ignored = plt.hist(s, 30, normed=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
# plt.show()