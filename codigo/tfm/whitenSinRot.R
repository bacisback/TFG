whitenSinRot <- function(data) {
  
  nseries <- ncol(data)    
  
  if (identical(round(cor(data), 6), diag(1, nseries))) {
    # already uncorrelated
    UU <- data
  } else {
    
    Sigma.X <- cov(data)
    VV <- eigen(Sigma.X, symmetric = TRUE)$vectors
    UU <- data %*% VV
  }
  return(UU)
} 