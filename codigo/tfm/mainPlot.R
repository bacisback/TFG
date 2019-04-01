mainPlot <- function(typ) {
  if(typ==1){ return(expression("Linear")) }
  if(typ==2){ return(expression("Parabolic")) }
  if(typ==3){ return(expression("Cubic")) }
  if(typ==4){ return(expression("Sin(4"*pi*"x)")) }
  if(typ==5){ return(expression("Sin(16"*pi*"x)")) }
  if(typ==6){ return(expression("Fourth root")) }
  if(typ==7){ return(expression("Circle")) }
  if(typ==8){ return(expression("Step")) }
  if(typ==9){ return(expression("2D cross gaussian")) }
}