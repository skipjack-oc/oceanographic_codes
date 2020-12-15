# Wavelet analysis using multivariate ENSO index
# Author: Claus Inck ft. Vladmyr Schlosser Mello
# OBS: we need to  

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
plot(wavelet, main = "MEI Wavelet Power Spectrum (Morlet)", bw = T, col.coi = 'red', col.sig = 'black')


# Reference
citation("biwavelet")

# end of the code !

