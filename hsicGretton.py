import numpy as np
from scipy.stats import gamma
def hsicTestGamma(x,y,alpha,params = [-1,-1]):
	m = len(x)
	if params[0] == -1:
		size1 = len(x)
		if size1 > 100:
			xmed = x[0:100]
			size1 = 100
		else:
			xmed = x
		G = (xmed*xmed)
		Q = np.repeat(G,size1,axis=1)
		R = np.repeat(G.reshape(-1,1),size1,axis=0)
		dists = Q + R -2*xmed*xmed.reshape(-1,1)
		dists = dists - np.tril(dists)
		dists = dists.reshape(size1*size1,1)
		params[0] = np.sqrt(0.5*np.median(dists[dists>0]))
	if params[1] == -1:
		size1 = len(y)
		
		if size1 > 100:
			ymed = y[0:100]
			size1 = 100
		else:
			ymed = y
		G = np.matrix((ymed*ymed),(size1,1))
		print(G)
		Q = np.repeat(np.matriG,size1,axis=1)
		R = np.repeat(G.reshape(-1,1),size1,axis=0)
		dists = Q + R -2*ymed*ymed.reshape(-1,1)
		dists = dists - np.tril(dists)
		dists = dists.reshape(size1*size1,1)
		params[1] = np.sqrt(0.5*np.median(dists[dists>0]))

	bone = np.ones((m,1))
	H = np.eye(m) -1./m*np.ones(m,m)

	K = rbf_dot(x,x,params[0])
	L = rbf_dot(y,y,params[1])

	Kc = H*K*H
	Lc = H*L*H

	testStat = 1./m * np.sum(Kc.T*Lc)

	varHsic = np.power((1./6 * Kc*Lc),2)
	varHsic = 1./m/(m-1) * (np.sum(varHsic) - np.sum(np.diag(varHsic)))
	varHsic = 72*(m-4)*(m-5)/m/(m-1)/(m-2)/(m-3) * varHsic

	K = K-np.diag(np.diag(K))
	L = L-np.diag(np.diag(L))

	muX = 1./m/(m-1)* bone.T *(K * bone)
	muY = 1./m/(m-1)* bone.T *(L * bone)

	mHsic = 1./m * ( 1+muX*muY -muX - muY)

	al = mHsic^2/varHsic
	bet = varHsic*m/mHsic

	thresh = 1-gamma.cdf(mHsic,al,bet)
	return [testStat,thresh]


def testHSIC_0(x,y,ruido):
	aux = y + np.random.normal(0,ruido,len(x))
	aux = (aux -np.mean(aux))/np.std(aux)
	[stat,value] = hsicTestGamma(x,y,0.05)
	print(stat,value)
	return value > 0.05

n = 100
x = np.random.rand(n)
y = np.log(x)
x = (x - np.mean(x))/np.std(x)


print("Res0\n",testHSIC_0(x,y,0))
print("Res1\n",testHSIC_0(x,y,1./9))
print("Res2\n",testHSIC_0(x,y,2./7))
print("Res3\n",testHSIC_0(x,y,3./5))

y = (x-0.5)**2
print("Res0\n",testHSIC_0(x,y,0))
print("Res1\n",testHSIC_0(x,y,1./9))
print("Res2\n",testHSIC_0(x,y,2./7))
print("Res3\n",testHSIC_0(x,y,3./5))