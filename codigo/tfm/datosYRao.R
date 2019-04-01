datosYRao <- function(n,typ,theta=0) {
  if (typ == 1) {
    #Gaussianas con corr 0.5
    corr = 0.5
    x = rnorm(n)#scale(rnorm(n))
    y = corr*x+sqrt(1-corr^2)*rnorm(n)#scale(corr*x+sqrt(1-corr^2)*rnorm(n))
  } else if (typ == 2) {
    #Gaussiana por uniformes
    x1 = rnorm(n)
    y1 = rnorm(n)
    z = runif(n,0,2)
    x = as.matrix(x1*z)
    y = as.matrix(y1*z)
  } else if (typ == 3) {
    #Mixtura de 3 gaussianas
    library(MASS)
    z1 = mvrnorm(n,c(0,0),matrix(c(1,0,0,1),2,2))
    z2 = mvrnorm(n,c(0,0),matrix(c(1,0.8,0.8,1),2,2))
    z3 = mvrnorm(n,c(0,0),matrix(c(1,-0.8,-0.8,1),2,2))
    ind = runif(n)
    z = (ind<=0.6)*z1 + (ind>0.6 & ind<=0.8)*z2 + (ind>0.8)*z3
    x = z[,1]
    y = z[,2]
  } else if (typ == 4) {
    #Gaussianas con ruido mult
    corr = 0.8
    x = scale(rnorm(n))
    y = scale(corr*x+sqrt(1-corr^2)*rnorm(n))
    x = x*rnorm(n)
    y = y*rnorm(n)
  } else if (typ == 5) {
    #Rotacion
    x = runif(n, -1, 1)
    y1 = runif(n,0.5,1)
    y2 = runif(n,-1,-0.5)
    ind = runif(n)
    y = (ind<=0.5)*y1 + (ind>0.5)*y2
    
    x = (x-mean(x))/sd(x)
    y = (y-mean(y))/sd(y)
    
    if (theta!=0){
      #Se rotan los datos
      R = matrix(c(cos(theta),sin(theta),-sin(theta),cos(theta)),2,2)
      data = R%*%rbind(x,y)
      x = data[1,]
      y = data[2,]
    }
  } 
  
  return(list(x=x,y=y))
} 