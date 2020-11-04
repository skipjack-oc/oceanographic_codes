"""
Created on Thu Oct 22 21:49:08 2020

@author: skipjack - Vladmyr Schlosser Mello ft. Vinicius Dionysio Alves

Principal Components Analisys, hipotetic data from trace metals.

data source: https://www.cambridge.org/br/academic/subjects/earth-and-environmental-science/oceanography-and-marine-science/modeling-methods-marine-science?format=HB&isbn=9780521867832

OBS: This code works, but... it's not ready. ;)
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load .mat files
f = sio.loadmat('TraceMeast.mat') #file?

# Construct Dataframe
trace_meast = DataFrame(np.array(f['X']), columns=['a', 'b', 'c', 'd', 'e', 'f']) 

# PCA Analisys
pca = PCA(n_components = 6)
pca.fit(trace_meast)

# Eigenvalues
autoval = np.round(pca.explained_variance_, 1)
print('Autovalores')
print(autoval)

# Eigenvectors
autovet = pca.components_
print('Autovetores')
print(autovet)


# Scaling data
trace_meast_std = StandardScaler().fit_transform(trace_meast)

# Creating covariance matrix ; "data" are columns from trace_meast_std
data = trace_meast_std.T
matriz_cov = np.cov(data)
print(matriz_cov)

# Calculating Eigenvalues and Eigenvectors for plotting
auto_vals, auto_vecs = np.linalg.eig(matriz_cov)
print('Autovetores \%s' %auto_vecs)
print('\n Autovalores \%s' %auto_vals )

# PoV - values percentage
PoV = (100 * auto_vals) / sum(auto_vals)

# Factorial load Ar from PC's
Ar = auto_vecs * (auto_vals**(0.5))

# Plot Factorial load
plt.plot(Ar)
plt.ylabel('Fator de Carregamento')
plt.legend()
plt.show()

