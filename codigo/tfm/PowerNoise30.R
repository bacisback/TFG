.libPaths(c( .libPaths(), "/usr/local/R/3.0.1/lib64/R/library") )
library(parallel)
source("mainPlot.R")


noiseLevel <- function(noiseArg) {
  .libPaths(c( .libPaths(), "/usr/local/R/3.0.1/lib64/R/library") )
  library(energy)
  source("datosY.R")
  source("whiten.R")
  source("whitenSinRot.R")
  source("ACI.R")
  source("gaussianHSIC.R")
  source("sigest.R")
  source("nGaussMMDenergy.R")
  source("rdc.R")
  
  set.seed(1)
  
  white = 1
  
  nsim=200  #500                         # The number of null datasets we use to estimate our rejection reject regions for an alternative with level 0.05
  nsim2=200   #500                        # Number of alternative datasets we use to estimate our power
  
  val.MMDmax = val.MMDmean = val.Emax = val.Emean = val.gaussmean = val.hsic = val.aci = val.dcor = val.rdc = rep(NA,nsim)              # Vectors holding the null "correlations" (for pearson, dcor and mic respectively) for each of the nsim null datasets at a given noise level
  
  val.MMDmax2 = val.MMDmean2 = val.Emean2 = val.Emax2 = val.gaussmean2 = val.hsic2 = val.aci2 = val.dcor2 = val.rdc2 =rep(NA,nsim2)              # Vectors holding the alternative "correlations" (for pearson, dcor and mic respectively) for each of the nsim2 alternative datasets at a given noise level
  
  MMDmax=MMDmean=Emean=Emax=gaussmean=hsic=aci=dcor=rdc=rep(NA,9)                # Arrays holding the estimated power for each of the "correlation" types, for each data type (linear, parabolic, etc...) with each noise level
  
  n=300                             # Number of data points per simulation
  
  rhoPoints = 26 #Sin coger el 0, da problemas con la escalon
  
  alphaE = alphaMMD = Inf
  
  k = 10 # Number of random projections for rdc method
  
  alpha = 0.05
  
  ##################################################################################
 
  if (white) {
    fich = paste0("ProcNoise-White", noiseArg, ".txt")
  } else {
    fich = paste0("ProcNoise", noiseArg, ".txt")
  }
  write("COMIENZO", fich)
  
  for(typ in 7:7){
    write(paste("Tipo: ", typ), fich, append = TRUE)

    ## This next loop simulates data under the null with the correct marginals (x is uniform, and y is a function of a uniform with gaussian noise)
    
    #Se ajusta el nivel de ruido para algunos datos
    noise = noiseArg
    if (typ == 3) noise = noiseArg*10
    if (typ == 4) noise = noiseArg*2
    if (typ == 7) noise = noiseArg/4
    if (typ == 8) noise = noiseArg*5
    
    #Calculamos las constantes de escala
#     while(alphaE==Inf || alphaMMD==Inf) {
#       x=runif(n)
#       data = datosY(x,typ)
#       x = scale(data$x)
#       y = scale(data$y + noise*rnorm(data$n))
#       dataC = nGaussMMDenergy(x,y,rhoPoints)
#       alphaMMD = mean(abs(dataC$diffG))/mean(abs(dataC$diffMMD))
#       alphaE = mean(abs(dataC$diffG))/mean(abs(dataC$diffE))
#     }    
    
    for(ii in 1:nsim){
      write(paste("Sim1 ", ii), fich, append = TRUE)
      
      x=runif(n)
      
      data = datosY(x,typ)
      x = data$x
      y = data$y
      
      # We resimulate x so that we have the null scenario
      if (typ == 9) {
        x <- rnorm(data$n)
      } else {x <- runif(n)} 
      
      #Standarize 
      x = scale(x)
      y = scale(y) + noise*rnorm(data$n)

      
      #Whitening
      if (white) {
        #dataW = whiten(cbind(x,y))$U
        dataW = whitenSinRot(cbind(x,y))
        x = scale(dataW[,1])
        y = scale(dataW[,2])
      }
             
      sigma = sigest(rbind(x,y),scaled=FALSE)[2]      #Median of |x-x'|^2
      if (sigma < 0.1) sigma = 0.1
      val.hsic[ii] = gaussianHSIC(x,y,sigma)          # Calculare HSIC
      val.dcor[ii] = dcov(x,y)                       # Calculate dcor
      val.aci[ii] = aci(x,y,sigma)                    # Calculate ACI
      s = c(sigest(x,scaled=FALSE)[2], sigest(y,scaled=FALSE)[2])
      val.rdc[ii] = rdc(x,y,k,s)
      
      dataG = nGaussMMDenergy(x,y,rhoPoints)
      alphaE = dataG$scaleE
      alphaMMD = dataG$scaleMMD
      
      val.gaussmean[ii] = mean(abs(dataG$diffG)) + dataG$LD      
      val.MMDmax[ii] = max(abs(alphaMMD*dataG$diffMMD)) + dataG$LD
      val.MMDmean[ii] = mean(abs(alphaMMD*dataG$diffMMD)) + dataG$LD
      val.Emax[ii] = max(abs(alphaE*dataG$diffE)) + dataG$LD
      val.Emean[ii] = mean(abs(alphaE*dataG$diffE)) + dataG$LD
    }
    
    ## Next we calculate our 9 rejection cutoffs
    cut.hsic=quantile(val.hsic,1-alpha)
    cut.dcor=quantile(val.dcor,1-alpha)
    cut.aci=quantile(val.aci,1-alpha)
    cut.rdc=quantile(val.rdc,1-alpha)
    cut.gaussmean=quantile(val.gaussmean,1-alpha)
    cut.MMDmax=quantile(val.MMDmax,1-alpha)
    cut.MMDmean=quantile(val.MMDmean,1-alpha)
    cut.Emax=quantile(val.Emax,1-alpha)
    cut.Emean=quantile(val.Emean,1-alpha)
    
    write("H1", fich, append = TRUE)
    
    ## Next we simulate the data again, this time under the alternative
    
    for(ii in 1:nsim2){
      write(paste("Sim2 ", ii), fich, append = TRUE)
      
      x=runif(n)
      
      data = datosY(x,typ)
      x = data$x
      y = data$y
      
      #Standarize
      x = scale(x)
      y = scale(y) + noise*rnorm(data$n)
      
      #Whitening
      if (white) {
        #dataW = whiten(cbind(x,y))$U
        dataW = whitenSinRot(cbind(x,y))
        x = scale(dataW[,1])
        y = scale(dataW[,2])
      }
      
      sigma = sigest(rbind(x,y),scaled=FALSE)[2]       #Median of |x-x'|^2
      if (sigma < 0.1) sigma = 0.1
      val.hsic2[ii] = gaussianHSIC(x,y,sigma)          # Calculare HSIC
      val.dcor2[ii] = dcov(x,y)                        # Calculate dcor
      val.aci2[ii] = aci(x,y,sigma)                    # Calculate ACI
      if (is.na(sigest(x,scaled=FALSE)[2]) || is.null(sigest(x,scaled=FALSE)[2]) || length(sigest(x,scaled=FALSE)[2])==0) {
        write(x,"Error.txt")
        write(y,"Error.txt",append=TRUE)
        write(s,"Error.txt",append=TRUE)
      }
      s = c(sigest(x,scaled=FALSE)[2], sigest(y,scaled=FALSE)[2])
      if (s[1] < 0.1) s[1] = 0.1
      if (s[2] < 0.1) s[2] = 0.1
      val.rdc2[ii] = rdc(x,y,k,s)

      dataG = nGaussMMDenergy(x,y,rhoPoints)
      alphaE = dataG$scaleE
      alphaMMD = dataG$scaleMMD
      
      val.gaussmean2[ii] = mean(abs(dataG$diffG)) + dataG$LD      
      val.MMDmax2[ii] = max(abs(alphaMMD*dataG$diffMMD)) + dataG$LD
      val.MMDmean2[ii] = mean(abs(alphaMMD*dataG$diffMMD)) + dataG$LD
      val.Emax2[ii] = max(abs(alphaE*dataG$diffE)) + dataG$LD
      val.Emean2[ii] = mean(abs(alphaE*dataG$diffE)) + dataG$LD
    }
    
    ## Now we estimate the power as the number of alternative statistics exceeding our estimated cutoffs
    
    hsic[typ] <- sum(val.hsic2 > cut.hsic)/nsim2
    dcor[typ] <- sum(val.dcor2 > cut.dcor)/nsim2
    aci[typ] <- sum(val.aci2 > cut.aci)/nsim2  
    rdc[typ] <- sum(val.rdc2 > cut.rdc)/nsim2  
    gaussmean[typ] <- sum(val.gaussmean2 > cut.gaussmean)/nsim2
    MMDmax[typ] <- sum(val.MMDmax2 > cut.MMDmax)/nsim2
    MMDmean[typ] <- sum(val.MMDmean2 > cut.MMDmean)/nsim2
    Emax[typ] <- sum(val.Emax2 > cut.Emax)/nsim2
    Emean[typ] <- sum(val.Emean2 > cut.Emean)/nsim2

    if (white) {
      fichTyp = paste0("Power",noiseArg,"-White",typ,".txt")
    } else {
      fichTyp = paste0("Power",noiseArg,"-",".txt")
    }
    write(rbind(hsic[typ],dcor[typ],aci[typ],gaussmean[typ],MMDmax[typ],MMDmean[typ],Emax[typ],Emean[typ],rdc[typ]), fichTyp)
  }
  
  #Se elimina el fichero de las iteraciones
  unlink(fich)

  return(rbind(hsic,dcor,aci,gaussmean,MMDmax,MMDmean,Emax,Emean,rdc))
}


#############################################################################


white = 1 #Cambiarlo también dentro de la función

num.meth = 9
num.data =1#9
num.noise <- 12#30                     # The number of different noise levels used
noise <- 3                          # A constant to determine the amount of noise

power.MMDmean = power.MMDmax = power.Emean = power.Emax = power.gaussmean = power.hsic=power.aci=power.dcor = power.rdc = array(NA,c(num.data,num.noise))                # Arrays holding the estimated power for each of the "correlation" types, for each data type (linear, parabolic, etc...) with each noise level

arg = noise *(seq(1,num.noise)/num.noise)

#cl = makeCluster(1)#num.noise)
powers = sapply(arg,noiseLevel)
#powers = parSapply(cl,arg, noiseLevel)
#stopCluster(cl)

#Se almacenan los datos
main = "Power Noise"
name = "PowerNoise30"

if (!white){
  nametxt = paste(name, ".txt", sep='')
  name = paste(name, ".pdf", sep='')
} else {
  nametxt = paste(name, "-White.txt", sep='')
  name = paste(name, "-White.pdf", sep='')
}

#write.table(powers, nametxt)

for (i in 1:num.data) {
  power.hsic[i,] = powers[1 + (i-1)*num.meth,]
  power.dcor[i,] = powers[2 + (i-1)*num.meth,]
  power.aci[i,] = powers[3 + (i-1)*num.meth,]
  power.gaussmean[i,] = powers[4 + (i-1)*num.meth,]
  power.MMDmax[i,] = powers[5 + (i-1)*num.meth,]
  power.MMDmean[i,] = powers[6 + (i-1)*num.meth,]
  power.Emax[i,] = powers[7 + (i-1)*num.meth,]
  power.Emean[i,] = powers[8 + (i-1)*num.meth,]
  power.rdc[i,] = powers[9 + (i-1)*num.meth,]
}

colors = rainbow(num.meth)

#save.image()
#pdf(name)

if (num.data == 1) {
  mainp = mainPlot(9)
  plot(arg, power.hsic, ylim = c(0,1), main = mainp, 
       xlab = "Noise level", ylab = "Power", pch = 1, col = colors[1], type = 'b')
  points(arg, power.dcor, pch = 2, col = colors[2], type = 'b')
  points(arg, power.aci, pch = 3, col = colors[3], type = 'b')
  points(arg, power.rdc, pch = 4, col = colors[4], type = 'b')
  points(arg, power.gaussmean, pch = 5, col = colors[5], type = 'b')
  points(arg, power.Emean, pch = 6, col = colors[6], type = 'b')
  points(arg, power.Emax, pch = 7, col = colors[7], type = 'b')
  points(arg, power.MMDmean, pch = 8, col = colors[8], type = 'b')
  points(arg, power.MMDmax, pch = 9, col = colors[9], type = 'b')
  legend("topright", c("HSIC","dCor","ACI","RDC","Imean","Emean","Emax","MMDmean","MMDmax"), pch=c(1,2,3,4,5,6,7,8,9), col=colors)  
} else {
  par(mfrow = c(3,3), cex = 0.45)
  for (i in 1:num.data) {
    mainp = mainPlot(i)
    plot(arg, power.hsic[i,], ylim = c(0,1), main = mainp, 
         xlab = "Noise level", ylab = "Power", pch = 1, col = colors[1], type = 'b')
    points(arg, power.dcor[i,], pch = 2, col = colors[2], type = 'b')
    points(arg, power.aci[i,], pch = 3, col = colors[3], type = 'b')
    points(arg, power.rdc[i,], pch = 4, col = colors[4], type = 'b')
    points(arg, power.gaussmean[i,], pch = 5, col = colors[5], type = 'b')
    points(arg, power.Emean[i,], pch = 6, col = colors[6], type = 'b')
    points(arg, power.Emax[i,], pch = 7, col = colors[7], type = 'b')
    points(arg, power.MMDmean[i,], pch = 8, col = colors[8], type = 'b')
    points(arg, power.MMDmax[i,], pch = 9, col = colors[9], type = 'b')

    if (i!=1 && i != 2 && i!=6 && i!=7) {
      legend("topright", c("HSIC","dCor","ACI","RDC","Imean","Emean","Emax","MMDmean","MMDmax"), pch=c(1,2,3,4,5,6,7,8,9), col=colors)
    }
    
    
#     if (i == 1 || i == 6 || i == 7) {
#       legend("bottomleft", c("HSIC","dCor","ACI","RDC","Imean","Emean","Emax","MMDmean","MMDmax"), pch=c(1,2,3,4,5,6,7,8,9), col=colors)
#     } else if (i != 2) {
#       legend("topright", c("HSIC","dCor","ACI","RDC","Imean","Emean","Emax","MMDmean","MMDmax"), pch=c(1,2,3,4,5,6,7,8,9), col=colors)
#     }
  }
}

#dev.off()
par(mfrow = c(1,1))

# #PARABOLIC
# i=2
# power.hsic = powers[1 + (i-1)*num.meth,]
# power.dcor = powers[2 + (i-1)*num.meth,]
# power.aci = powers[3 + (i-1)*num.meth,]
# power.gaussmean = powers[4 + (i-1)*num.meth,]
# power.MMDmax = powers[5 + (i-1)*num.meth,]
# power.MMDmean = powers[6 + (i-1)*num.meth,]
# power.Emax = powers[7 + (i-1)*num.meth,]
# power.Emean = powers[8 + (i-1)*num.meth,]
# 
# #CIRCLE
# power.hsic = powers[49,]
# power.dcor = powers[50,]
# power.aci = powers[51,]
# power.gaussmean = powers[52,]
# power.MMDmax = powers[53,]
# power.MMDmean = powers[54,]
# power.Emax = powers[55,]
# power.Emean = powers[56,]

# #Fourth root
# i=6
# power.hsic = powers[1 + (i-1)*num.meth,]
# power.dcor = powers[2 + (i-1)*num.meth,]
# power.aci = powers[3 + (i-1)*num.meth,]
# power.gaussmean = powers[4 + (i-1)*num.meth,]
# power.MMDmax = powers[5 + (i-1)*num.meth,]
# power.MMDmean = powers[6 + (i-1)*num.meth,]
# power.Emax = powers[7 + (i-1)*num.meth,]
# power.Emean = powers[8 + (i-1)*num.meth,]


# 
# power.hsic = resCCC[1,]
# power.dcor = resCCC[2,]
# power.aci = resCCC[3,]
# power.gaussmean = resCCC[4,]
# power.Emean = resCCC[5,]
# power.Emax = resCCC[6,]
# power.MMDmean = resCCC[7,]
# power.MMDmax = resCCC[8,]
