# The following function computes the cross information potential between
#               two vectors X and Y using a double exponential kernel
#
# Input:   X and Y real valued COLUMN vectors of same length
#  			kSize is a scalar for the kernel size, i.e. K(x,y) =  exp(-|x-y|/kSize)
#
# Output: cip contains the cross information potential
#
# Default:  kSize = 1.
#
# Author: IL "Memming" Park (memming@cnel.ufl.edu)	Date: 15.07.2009
# Translation to R: Beatriz Bueno (bea.bueno@estudiante.uam.es) Date: 11.12.2014 
cipexp <- function(X, Y, kSize=1) {
  X = X/kSize; Y = Y/kSize
  
  X_sort = as.matrix(sort(X)); Y_sort = as.matrix(sort(Y))
  
  X_pos_exp = exp(X_sort); X_neg_exp = exp(-X_sort)
  Y_pos_exp = exp(Y_sort); Y_neg_exp = exp(-Y_sort)
  
  Y_pos_cum_sum_exp  = cumsum(Y_pos_exp)
  Y_neg_cum_sum_exp = apply(as.matrix(cumsum(apply(Y_neg_exp,2,rev))),2,rev)
  
  Y_sort = c(Y_sort, Inf)
  cip = 0
  yidx = 0 # no y is smaller than x yet
  for (xidx in 1:length(X)) {
    x = X_sort[xidx]
    
    # at the end of the next Y_sort(yidx) >= X_sort(xidx)
    while(Y_sort[yidx+1] <= x)
      yidx = yidx + 1
    
    if (yidx == 0) { 
      cip = cip + Y_neg_cum_sum_exp[1] * X_pos_exp[xidx]
    } else if (yidx == length(Y)) {
      cip = cip + Y_pos_cum_sum_exp[length(Y_pos_cum_sum_exp)] * X_neg_exp[xidx] 
    } else {
      cip = cip + Y_pos_cum_sum_exp[yidx] * X_neg_exp[xidx] + Y_neg_cum_sum_exp[yidx+1] * X_pos_exp[xidx]
    }
  }
  return(cip / (length(X) * length(Y)))
}



# The following function computes the parametric centered correntropy
#               between two vectors X and Y using double exponential kernel
#
# Input:   X and Y are real valued COLUMN vectors of SAME length
#  			kSize is a scalar for the kernel size, i.e. K(x,y) = exp(-|x-y|/kSize),
#				param is a [2x1] vector containing the coefficients a and b in ORDER.
#
# Output: corren contains the parametric centered correntropy
#
# Default:  param = [a,b] = [1,0] and kSize = 1.
#
# Comments: The code uses cipexp
#
# Author: Sohan Seth (sohan@cnel.ufl.edu)	Date: 15.07.2009
# Translation to R: Beatriz Bueno (bea.bueno@estudiante.uam.es) Date: 11.12.2014 
aciexp <- function(X, Y, kSize=1, param=c(1,0)) {
  
  n = nrow(X)
  a = param[1]; b = param[2];
  
  X = a*X+b;
  
  return((1/n)*sum(exp(-abs(X - Y)/kSize)) - cipexp(X,Y,kSize))
}


aci <- function(X, Y, kSize=1) {
  
  return(max(abs(aciexp(X,Y, kSize)), abs(aciexp(X,Y,kSize,c(-1,0)))))
}


eci <- function(X, Y, kSize=1) {
  a = tan(seq(0,pi,length=41))
  b = seq(-2,2,length=9)
  
  n = length(a); m = length(b)
  eci.val = matrix(rep(0,n*m),n,m)
  for (i in 1:n) {
    for (j in 1:m) {
      eci.val[i,j] = aciexp(X, Y, kSize, c(a[i],b[j]))
    }
  }
  
  return(max(eci.val))
}