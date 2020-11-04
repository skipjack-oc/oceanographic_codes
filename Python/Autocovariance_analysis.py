# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: skipjack - Vladmyr Schlosser Mello
This is a temporary script file.

Adapted from: Vinicius Dionysio Alves

Autocovariance Analysis from tide data.
data source: http://www.cambridge.org/glover
"""

# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.tsa.stattools as st

# Load .dat files
f = np.loadtxt('woods_hole_tides_2005_05.dat')

# Construct Dataframe
col = ['ID', 'year', 'month', 'day', 'hour', 'minute', 'predMLLW', 'verMLLW']
data = pd.DataFrame(f, columns=col)

# Format datetime for Y-M-D H:m
date = pd.DataFrame(data[col[1:6]])
date = pd.to_datetime(date, format='%Y-%M-%D %H:%m')

# Reconstruct dataframe with new datetime
copy = data.copy()  # secure copy
for i in col[1:6]:  # remove time columns
    data.drop(i, axis=1, inplace=True)
data.insert(1, 'date', date)  # add single column datetime

# New column with datetime for plotting
data['diahora'] = data['date'].map(lambda x: x.strftime('%d %H:%m'))

# Sea level (2015/may) time serie plot  
plt.plot(data['diahora'], data['verMLLW'])
plt.xlim(data['diahora'][0], data['diahora'][7439])
plt.xticks(data['diahora'][0::240], rotation=90)
plt.title('Sea level (2015/may) time serie - nver. MLLW - May 2005')
plt.ylabel('Tide level (m)')
plt.grid()
plt.show()

# Autocovariance calculating
N = len(data['verMLLW'])
a_cov = st.acovf(data['verMLLW'], unbiased=True)
a_cov = np.delete(a_cov, N-1) # Removing last element

# Autocovariance plots 
x = np.arange(0, N-1)/10  # Lag for the plots

# Total
plt.subplot(211)
plt.plot(x, a_cov)
plt.xlim(x[0], x[N-2]) # Minus 2 because 0 index
plt.xticks(x[::400])
plt.ylabel('Autocovariance [m²]')
plt.title('Autocovariance Function ver. MLLW \ Total ')

# 0 - 24h Lag
plt.subplot(223)
plt.plot(x[:241], a_cov[:241])
plt.xlim(x[0], x[240])
plt.xticks(x[:241:30])
plt.ylabel('Autocovariance [m²]')
plt.xlabel('lag')
plt.title('0 - 24 hours')

# 0 - 150h Lag
plt.subplot(224)
plt.plot(x[:1441], a_cov[:1441])
plt.title('0 - 144 hours')
plt.xlabel('lag')
plt.xlim(x[0], x[1440])
plt.xticks(x[:1441:120])
plt.show()

# Estimate decorrelation time
# Hook to find first zero
i_tdec = 0  # Decorrelation time index
for c in range(0, N):
    if(a_cov[c] <= 0):
        i_tdec = c
        break

# Autocovariance function with Decorrelation time 0 -24h lag
plt.plot(x[:241], a_cov[:241])
plt.xlim(x[0], x[240])
plt.xticks(x[:241:40])
plt.annotate('decorrelation\ntime',xy=[x[i_tdec],0],xytext=[6,0.02],arrowprops=dict(facecolor='black'))
plt.ylabel('Autocovariância [m²]')
plt.axhline(0, color='black')
plt.axvline(4, color='black')
plt.title('Autocovariance function \nlag 0 - 24 hours')
plt.xlabel('lag')
plt.show()

# Random data Autocovariance
rand_data = np.random.randn(7440)
a_cov_rand = st.acovf(rand_data, unbiased=True)
a_cov_rand = np.delete(a_cov_rand,N-1)

# Random data Autocovariance seie plot
plt.subplot(211)
plt.plot(rand_data)
plt.xlim(0, len(rand_data))
plt.title('Random data Series')

# Random data Autocovariance function plot
plt.plot(a_cov_rand)
plt.title('Autocovariance function from random data')
plt.axhline(0, color='black')
plt.ylabel('Autocovariance')
plt.xlabel('lag')
plt.show()

# Autocorrelation and critical limits
# Autocorrelation
a_cor = a_cov/(np.std(data['verMLLW'])**2)

# Random data
a_cor_rand = a_cov_rand/(np.std(rand_data)**2)

# Autocorrelation functions for real data
plt.subplot(211)
plt.plot(x,a_cor)
plt.xlim(x[0],x[N-2])
plt.title('Autocorrelation Function from real data')
plt.ylabel('Autocorrelation')

# Autocorrelation functions for random data
plt.plot(x, a_cor_rand)
plt.xlim(x[0],x[N-2])
plt.title('Autocorrelation Function from random data')
plt.ylabel('Autocorrelation')
plt.xlabel('lag - hours')
plt.show()

# Autocorrelation functions with critical limits plot
rn = 1.96/np.sqrt(N-(x*10)-3)

# Real data plot
plt.subplot(211)
plt.plot(a_cor)
plt.plot(rn,color='black',linestyle='dashed')
plt.plot(-rn, color='black',linestyle='dashed')
plt.title('Função Autocorrelação dos dados reais')
plt.ylabel('Autocorrelação')

# Random data plot
plt.subplot(212)
plt.plot(a_cor_rand)
plt.plot(rn,color='black',linestyle='dashed')
plt.plot(-rn,color='black',linestyle='dashed')
plt.title('Função Autocorrelação dos dados aleatórios')
plt.ylabel('Autocorrelação')
plt.show()
