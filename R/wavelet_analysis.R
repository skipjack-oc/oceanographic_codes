# Wavelet analysis using multivariate ENSO index
# Author: Claus Inck ft. Vladmyr Schlosser Mello
# OBS: we need to correct, date axis!!
# Data Source: http://paos.colorado.edu/research/wavelets/software.html

# Set working directory
setwd("/home/skipjack/R")

# Install packages
install.packages("biwavelet")

# Import libraries
library(R.matlab)

# Load data
mei <- read.csv("mei_1950-2015.txt", sep = "")

# Time serie vector
timeserie <- vector()
count <- 1
for(c in 1:nrow(mei)){
 for(r in 2:ncol(mei)){
  timeserie[count] <- mei[c,r]
  count <- count+1}
 }

# Time serie plot
plot(ts(timeserie), main = "Mulrivariate ENSO Index", ylab= "(ºC)")

# Wavelet analysis
wavelet <- biwavelet::wt(cbind(1:791, timeserie))

# Wavelet plot
plot(wavelet, main = "MEI Wavelet Power Spectrum (Morlet)") 


# Reference
citation("biwavelet")

## To cite biwavelet in publications use:
## 
##   Tarik C. Gouhier, Aslak Grinsted, Viliam Simko (2019). R package
##   biwavelet: Conduct Univariate and Bivariate Wavelet Analyses (Version
##   0.20.19). Available from https://github.com/tgouhier/biwavelet
## 
## A BibTeX entry for LaTeX users is
## 
##   @Manual{,
##     title = {R package {biwavelet}: Conduct Univariate and Bivariate Wavelet Analyses},
##     author = {Tarik C. Gouhier and Aslak Grinsted and Viliam Simko},
##     year = {2019},
##     note = {(Version 0.20.19)},
##     url = {https://github.com/tgouhier/biwavelet},
##   }

# end of the code !

