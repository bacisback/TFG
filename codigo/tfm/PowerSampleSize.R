#.libPaths(c( .libPaths(), "/usr/local/R/3.0.1/lib64/R/library") )
library(parallel)
source("mainPlot.R")


sampleSize <- function(n) {
#  .libPaths(c( .libPaths(), "/usr/local/R/3.0.1/lib64/R/library") )
  library(energy)
  source("datosYRao.R")
  source("whiten.R")
  source("whitenSinRot.R")
  source("ACI.R")
  source("gaussianHSIC.R")
  source("sigest.R")
  source("nGaussMMDenergy.R")
  source("rdc.R")
  
  set.seed(1)
  
  white = 1
  
  nsim=500                         # The number of null datasets we use to estimate our rejection reject regions for an alternative with level 0.05
  nsim2=500                        # Number of alternative datasets we use to estimate our power
  
  val.MMDmax = val.MMDmean = val.Emax = val.Emean = val.gaussmean = val.hsic=val.aci=val.dcor=val.rdc=rep(NA,nsim)              # Vectors holding the null "correlations" (for pearson, dcor and mic respectively) for each of the nsim null datasets at a given noise level
  
  val.MMDmax2 = val.MMDmean2 = val.Emean2 = val.Emax2 = val.gaussmean2 = val.hsic2=val.aci2=val.dcor2=val.rdc2=rep(NA,nsim2)              # Vectors holding the alternative "correlations" (for pearson, dcor and mic respectively) for each of the nsim2 alternative datasets at a given noise level
  
  MMDmax=MMDmean=Emean=Emax=gaussmean=hsic=aci=dcor=rdc=rep(NA,4)                # Arrays holding the estimated power for each of the "correlation" types, for each data type (linear, parabolic, etc...) with each noise level
  
  rhoPoints = 26 #Sin coger el 0, da problemas con la escalon
  
  alphaE = alphaMMD = Inf
  
  k = 10 # Number of random projections for rdc method
  
  alpha = 0.05 #Signification level
  
  ###############################################################################
  
  if (white) {
    fich = paste0("ProcSample-White", n, ".txt")
  } else {
    fich = paste0("ProcSample", n, ".txt")
  }
  write("COMIENZO", fich)
  
  for(typ in 1:4){
    write(paste("Tipo: ", typ), fich, append = TRUE)
    
    #Calculamos las constantes de escala
#     while(alphaE==Inf || alphaMMD==Inf) {
#       data = datosYRao(n,typ)
#       x = scale(data$x)
#       y = scale(data$y)
#       dataC = nGaussMMDenergy(x,y,rhoPoints)
#       alphaMMD = mean(abs(dataC$diffG))/mean(abs(dataC$diffMMD))
#       alphaE = mean(abs(dataC$diffG))/mean(abs(dataC$diffE))
#     }    
    
    for(ii in 1:nsim){
      write(paste("Sim1 ", ii), fich, append = TRUE)
      
      data = datosYRao(n,typ)
      x = data$x
      y = data$y
      
      # We resimulate x so that we have the null scenario
      x = datosYRao(n,typ)$x      
      
      #Whitening
      if (white) {
        dataW = whiten(cbind(x,y))$U
        #dataW = whitenSinRot(cbind(x,y))
        x = dataW[,1]
        y = dataW[,2]
      }
      
      #Standarize 
      x = scale(x)
      y = scale(y)
      
      sigma = sigest(rbind(x,y),scaled=FALSE)[2]      #Median of |x-x'|^2
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
    
    ## Next we calculate our 3 rejection cutoffs
    cut.hsic=quantile(val.hsic,1-alpha)
    cut.dcor=quantile(val.dcor,1-alpha)
    cut.aci=quantile(val.aci,1-alpha)
    cut.rdc=quantile(val.rdc,1-alpha)
    cut.gaussmean=quantile(val.gaussmean,1-alpha)
    cut.MMDmax=quantile(val.MMDmax,1-alpha)
    cut.MMDmean=quantile(val.MMDmean,1-alpha)
    cut.Emax=quantile(val.Emax,1-alpha)
    cut.Emean=quantile(val.Emean,1-alpha)
    
    
    ## Next we simulate the data again, this time under the alternative
    
    for(ii in 1:nsim2){
      write(paste("Sim2 ", ii), fich, append = TRUE)
      
      data = datosYRao(n,typ)
      x = data$x
      y = data$y
      
      #Whitening
      if (white) {
        dataW = whiten(cbind(x,y))$U
        #dataW = whitenSinRot(cbind(x,y))
        x = dataW[,1]
        y = dataW[,2]
      }
      
      #Standarize
      x = scale(x)
      y = scale(y)
      
      sigma = sigest(rbind(x,y),scaled=FALSE)[2] #Median of |x-x'|^2
      val.hsic2[ii] = gaussianHSIC(x,y,sigma)          # Calculare HSIC
      val.dcor2[ii] = dcov(x,y)                        # Calculate dcor
      val.aci2[ii] = aci(x,y,sigma)                    # Calculate ACI
      s = c(sigest(x,scaled=FALSE)[2], sigest(y,scaled=FALSE)[2])
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
  }
  
  #Se elimina el fichero de las iteraciones
  unlink(fich)
  
  return(rbind(hsic,dcor,aci,gaussmean,MMDmax,MMDmean,Emax,Emean,rdc))
}


#############################################################################


white = 1 #Cambiarlo también dentro de la función

num.meth = 9
num.data = 4
arg = c(25,50,100,150,200)
#arg = c(25,50,100,150)
#arg = c(25,50,100)
#arg = c(150)

power.MMDmean = power.MMDmax = power.Emean = power.Emax = power.gaussmean = power.hsic=power.aci=power.dcor= power.rdc = array(NA,c(num.data,length(arg)))                # Arrays holding the estimated power for each of the "correlation" types, for each data type (linear, parabolic, etc...) with each noise level

#cl = makeCluster(4)#length(arg))
powers = sapply(arg,sampleSize)
#powers = parSapply(cl,arg, sampleSize)
#stopCluster(cl)

#Se almacenan los datos
main = "Power Sample Size"
name = "PowerSampleSize-Definitivo"

if (!white){
  nametxt = paste(name, ".txt", sep='')
  name = paste(name, ".pdf", sep='')
} else {
  nametxt = paste(name, "-White.txt", sep='')
  name = paste(name, "-White.pdf", sep='')
}

write.table(powers, nametxt)

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

names = c("Gaussian corr 0.5", "Gaussians multiply uniform", "Mixture 3 gaussians", "Gaussian multiplicative noise")

colors = rainbow(num.meth)

save.image()
pdf(name)

if (num.data == 1) {
  mainp = names[1]
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
  par(mfrow = c(2,2), cex = 0.45)
  for (i in 1:num.data) {
    mainp = names[i]
    plot(arg, power.hsic[i,], ylim = c(0,1), main = mainp, 
         xlab = "Sample size", ylab = "Power", pch = 1, col = colors[1], type = 'b')
    points(arg, power.dcor[i,], pch = 2, col = colors[2], type = 'b')
    points(arg, power.aci[i,], pch = 3, col = colors[3], type = 'b')
    points(arg, power.rdc[i,], pch = 4, col = colors[4], type = 'b')
    points(arg, power.gaussmean[i,], pch = 5, col = colors[5], type = 'b')
    points(arg, power.Emean[i,], pch = 6, col = colors[6], type = 'b')
    points(arg, power.Emax[i,], pch = 7, col = colors[7], type = 'b')
    points(arg, power.MMDmean[i,], pch = 8, col = colors[8], type = 'b')
    points(arg, power.MMDmax[i,], pch = 9, col = colors[9], type = 'b')
    legend("topright", c("HSIC","dCor","ACI","RDC","Imean","Emean","Emax","MMDmean","MMDmax"), pch=c(1,2,3,4,5,6,7,8,9), col=colors)
  }
}

dev.off()
par(mfrow = c(1,1))
