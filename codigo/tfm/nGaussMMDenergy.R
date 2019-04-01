library(energy)
source("mmdStandardGaussian.R")

h_normal <- function(Z) {entropy = 0.5*log(2*pi*exp(1)*var(Z))}

h <- function(z) {
  z = as.matrix(z)
  euler_mascheroni = 0.57721566490153286060651209008240243104215933593992
  n = length(z)
  return (sum(log(n*diff(apply(z,2,sort))))/n + euler_mascheroni)
}

nGaussMMDenergy <- function(x,y, rhoPoints=101) {
  rho = seq(-1,1,length=rhoPoints)
  
  G = Gperm = MMD = MMDperm = E = Eperm = seq(0, rhoPoints-1)
  yperm = y[sample(length(y))]
  Cxy = cor(x,y)
  Cxyperm = 0#cor(x,yperm)
  
  scaleE = scaleMMD = rep(NA,rhoPoints)
   
  for (r in 1:rhoPoints) {
    w = rho[r]*x + sqrt(1-rho[r]^2)*y #Descomposicion para crear variables con correlacion
    wperm = rho[r]*x + sqrt(1-rho[r]^2)*yperm
    #G[r] = abs(h(wperm) - h(w) + 0.5*log(1+2*rho[r]*sqrt(1-rho[r]^2)*Cxy))
    #Gperm[r] = 0
    G[r] = log(2*pi*exp(1)*(1+2*rho[r]*sqrt(1-rho[r]^2)*Cxy))/2 - h(w)
    Gperm[r] = log(2*pi*exp(1)*(1+2*rho[r]*sqrt(1-rho[r]^2)*Cxyperm))/2 - h(wperm)
    
    MMD[r] = mmdStandardGaussian(scale(w), 1)$MMD
    MMDperm[r] = mmdStandardGaussian(scale(wperm), 1)$MMD
    E[r] = normal.e(scale(w)) 
    Eperm[r] = normal.e(scale(wperm))
    
    scaleE[r]= abs((h_normal(w) - h(w))/E[r])
    scaleMMD[r]= abs((h_normal(w) - h(w))/MMD[r])
  }
  LD = -log(1-abs(det(as.matrix(Cxy))))/2 #Mutual information
  
  return(list(diffG = (G-Gperm), diffMMD = (MMD-MMDperm), diffE = (E-Eperm), LD=LD, scaleE = median(scaleE), scaleMMD = median(scaleMMD)))
}