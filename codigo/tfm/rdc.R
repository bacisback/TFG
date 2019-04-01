rdc <- function(x,y,k,s) {
  x <- cbind(apply(as.matrix(x),2,function(u) ecdf(u)(u)),1)
  y <- cbind(apply(as.matrix(y),2,function(u) ecdf(u)(u)),1)
  wx <- matrix(rnorm(ncol(x)*k,0,s),ncol(x),k)
  wy <- matrix(rnorm(ncol(y)*k,0,s),ncol(y),k)
  cancor(cbind(cos(x%*%wx),sin(x%*%wx)), cbind(cos(y%*%wy),sin(y%*%wy)))$cor[1]
}