datosY <- function(xu, typ) {
  x = xu
  n = length(x)
  
  if(typ==1){
    y=xu
  }
  #parabolic
  if(typ==2){
    y=4*(xu-.5)^2
  }
  #cubic
  if(typ==3){
    y=128*(xu-1/3)^3-48*(xu-1/3)^3-12*(xu-1/3)
  }
  #sin
  if(typ==4){
    y=sin(4*pi*xu)
  }
  #their sine
  if(typ==5){
    y=sin(16*pi*xu)
  }
  #x^(1/4) 
  if(typ==6){
    y=xu^(1/4)
  }
  #circle
  if(typ==7){
    y=(2*rbinom(n,1,0.5)-1) * (sqrt(1 - (2*xu - 1)^2)) 
  }
  #step function
  if(typ==8){
    #y = (xu > 0.5)
    b = 10000
    y = 1.0/(1.0 + exp(-b*(xu-0.5)))
  }
  #2d-cross gaussian
  if(typ==9){
    library(matlab)
    C = 2*pi*1.85
    f_ref <- function(x,y) {return(exp(-(x*x+y*y)/2.0)/(2.0*pi))}
    aux_f <- function(x,y) {return(x*y*(x*x - y*y))}
    f     <- function(x,y) {return(f_ref(x,y)*(1+C*aux_f(x,y)*f_ref(x,y)))}
    A = 2
    N = ceil(n*A)
    x = matrix(rnorm(N), N, 1) 
    y = matrix(rnorm(N), N, 1)   # X,Y ~ f_ref (2D standard normal)
    U = A*f_ref(x,y)*matrix(runif(N), N, 1) 
    indices = U < f(x,y)         # reject/include sample
    x = x[indices]
    y = y[indices] 
  }
  
  return(list(x=x, y=y, n=length(x)))
}