import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from scipy.stats import rankdata, chi2
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
def rdc(x,y,k=7,s=0.2,n=5):
    if n > 1:
        values = []
        ks = []
        for i in range(n):
        	print(i,"   ",end=" ")
        	try:
        		ro,ko = rdc(x, y, k, s, 1)
        		values.append(ro)
        		ks.append(ko)
        		print(ro, end ="\r")
        	except np.linalg.linalg.LinAlgError: pass
        return values,ks
    lx = len(x)
    ly = len(y)
    x = np.concatenate((ecdf(x).reshape(-1,1),
        np.ones(lx).reshape(-1,1)),axis=1)
    y = np.concatenate((ecdf(y).reshape(-1,1),
        np.ones(ly).reshape(-1,1)),axis=1)
    
    nx = x.shape[1]
    ny = y.shape[1]
    #print(nx,k)
    wx =  np.random.normal(0,s,nx*k).reshape(nx,k)
    wy =  np.random.normal(0,s,ny*k).reshape(ny,k)
    wxs = np.matmul(x,wx)
    wys = np.matmul(y,wy)
    fX = np.sin(wxs)
    fY = np.sin(wys)
    [res,k] = cancor(fX,fY,k)
    return res,k
    wxs = np.concatenate((np.cos(wxs),np.sin(wxs)),axis=1)
    wys = np.concatenate((np.cos(wys),np.sin(wys)),axis=1)
    #d = rcancor(wxs,wys)
    #print("mio",d)
    cca = CCA(n_components=1)
    cca.fit(wxs,wys)

    xc,yc = cca.transform(wxs,wys)
    #result1 = cca.score(xc,yc)
    result2 = np.corrcoef(xc.T,yc.T)[0,1]
    #print(result1,result2)
    return result2, k
def cancor(x,y,k):
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

    return np.sqrt(np.max(eigs)),k
def testRDC(x,y,k=7,n=10):
	#Under H0: two sets are uncorrelated:
	#our statistic will follow a xisquared distribution
	power = []
	for _ in range(5):
		ros,ks = rdc(x,y,k,n=n)
		count = 0.0
		for i in range(len(ros)):
			if(ros[i] <1):
				statistic = ((2*ks[i] +3)/2 - n) * np.log(1.0-ros[i]**2)
				p = (1- chi2.cdf(statistic,df=ks[i]**2))
				#print (p,end="\r")
			else:
				p = 0
			if p < 0.05:
				count +=1.0

		power.append(count* 1./len(ros))
	return np.median(power)