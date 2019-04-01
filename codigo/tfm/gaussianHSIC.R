gaussianHSIC <- function(x,y,s) {
  m = length(x)
  K = matrix(rep(0,m*m),m,m)
  L = K
  
  for (i in 1:(m-1)) {
    xi = rep(x[i],m-i)
    xj = x[(i+1):m]
    K[i,(i+1):m] = exp(-s*s*(xi-xj)*(xi-xj))
    
    yi = rep(y[i],m-i)
    yj = y[(i+1):m]
    L[i,(i+1):m] = exp(-s*s*(yi-yj)*t(yi-yj))
  }
  
  K = K + t(K) + diag(m)
  L = L + t(L) + diag(m)

  H = diag(m) - matrix(rep(1/m,m*m),m,m)
  
  return(sum(diag(K%*%H%*%L%*%H)))
}