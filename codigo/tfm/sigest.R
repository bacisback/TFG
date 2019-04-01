## sigma estimation for RBF kernels
## author: alexandros

sigest <- function (x,
                    frac = 0.5,
                    scaled    = TRUE,
                    na.action = na.omit)
{
  x <- na.action(x)
  
  if (length(scaled) == 1)
    scaled <- rep(scaled, ncol(x))
  if (any(scaled)) {
    co <- !apply(x[,scaled, drop = FALSE], 2, var)
    if (any(co)) {
      scaled <- rep(FALSE, ncol(x))
      warning(paste("Variable(s)",
                    paste("`",colnames(x[,scaled, drop = FALSE])[co],
                          "'", sep="", collapse=" and "),
                    "constant. Cannot scale data.")
      )
    } else {
      xtmp <- scale(x[,scaled])
      x[,scaled] <- xtmp
    }
  }
  
  m <- dim(x)[1]
  n <- floor(frac*m)
  index <- sample(1:m, n, replace = TRUE)
  index2 <- sample(1:m, n, replace = TRUE)
  temp <- x[index,, drop=FALSE] - x[index2,,drop=FALSE]
  dist <- rowSums(temp^2)
  srange <- 1/quantile(dist[dist!=0],probs=c(0.9,0.5,0.1))
  
  return(srange)
}
