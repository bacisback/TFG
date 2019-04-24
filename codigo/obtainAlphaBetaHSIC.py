import numpy as np
def alpha_beta_hsic(x,y):
  params = [-1,-1]
  m = x.size
  if x.ndim == 1:
    x = x.reshape((-1,1))
  if y.ndim == 1:
    y = y.reshape((-1,1))
  """
  Sacamos los kernels de X e Y
  """
  if params[0] == -1:
    size1 = len(x)
    if size1 > 100:
      xmed = x[0:100]
      size1 = 100
    else:
      xmed = x
    G = np.sum(xmed*xmed,1)
    Q = np.repeat(G,size1).reshape(size1,size1)
    R = Q.T
    dists = Q + R -2*np.dot(xmed,xmed.T)
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
    G = np.sum(ymed*ymed,1)
    Q = np.repeat(G,size1).reshape(size1,size1)
    R = Q.T
    dists = Q + R -2*np.dot(ymed,ymed.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(size1*size1,1)
    params[1] = np.sqrt(0.5*np.median(dists[dists>0]))


  bone = np.ones((m,1))

  H = np.eye(m) -(1./m)*np.ones((m,m))

  K = rbf_dot(x,x,params[0])
  L = rbf_dot(y,y,params[1])

  Kc = (H.dot(K)).dot(H)
  Lc = (H.dot(L)).dot(H)

  testStat = 1./m * np.sum(Kc.T*Lc)

  varHsic = np.power((1./6 * Kc*Lc),2)
  varHsic = 1./m/(m-1) * (np.sum(varHsic) - np.sum(np.diag(varHsic)))
  varHsic = 72*(m-4)*(m-5)/m/(m-1)/(m-2)/(m-3) * varHsic

  K = K-np.diag(np.diag(K))
  L = L-np.diag(np.diag(L))
  
  
  
  muX = 1./m/(m-1)* bone.T.dot(K.dot(bone))
  muY = 1./m/(m-1)* bone.T.dot(L.dot(bone))


  mHsic = 1./m * ( 1+muX*muY -muX - muY)

  

  al = np.power(mHsic,2)*1./varHsic
  bet = varHsic*m*1./mHsic
  return[al,bet]

"""
Funcion auxiliar para sacar el producto interno
"""
def rbf_dot(patterns1,patterns2,deg,Flag = True):
  size1 = len(patterns1)
  size2 = len(patterns2)

  G = np.sum(patterns1*patterns1,1)
  Q = np.repeat(G,size1).reshape(size1,size1)
  if Flag:
    R = Q.T
    H = Q+R-2*np.dot(patterns1,patterns2.T)
    H = np.exp(-H/2./(deg**2))
    return H

  H = patterns2*patterns2
  R = np.repeat(H.T,size1).reshape(size1,size1)
  H = Q+R-2*patterns1*patterns2.T
  H = np.exp(-H/2./(deg**2))
  return H


alfa = np.zeros(1000)
beta = np.zeros(1000)
for size in [50,100,150,200,500,1000]:
  for i in range(1000):
    x, y = np.random.multivariate_normal(np.ones(2), np.eye(2), size).T
    alfa[i],beta[i] = alpha_beta_hsic(x,y)
  np.savetxt("datos/histogramas/alfa,beta-"+str(size)+".txt",np.stack((alfa,beta)),delimiter="\t")

