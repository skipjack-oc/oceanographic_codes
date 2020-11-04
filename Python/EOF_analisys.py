#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
!!!! NOT READY !!!!!

Created on Thu Oct 22 15:20:17 2020

@author: skipjack

Adaptado de: Vinicius Dyonisio Alves

Resolução do exercício 2 da Lista 3
Análise de EOF's, série temporal de dados artificiais de altura do nı́vel do
mar
"""
# !! RELER AULA EOF
# !!CONSERTAR PLOTS!!

# Carregar bibliotecas
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pandas import DataFrame
from eofs.standard import Eof # !! PESQUISAR MELHOR !!

# Carregar arquivo .mat
f = sio.loadmat('HypoGyre.mat')
Z = f['Z'] # relação entre f e Z?

# Analisar PC's
# Criar Solver
solver = Eof(Z)

# Retornar as PC's
# !!CHECAR PADRONIZAÇÃO E NORMALIZAÇÃO!!
pcs = solver.pcs()

# Ajustar os dados para uma matriz 2D ; [i (tempo) x j (altura)]
Z = np.transpose(Z).real 
Z = np.reshape(Z, [24, 900]) 
Z = DataFrame(Z)
lon = f['x'].squeeze()
lat = f['y'].squeeze()

# Matriz de Covariância
Z_cov = Z.cov()

# Calcular Autovalores e Autovetores
# !!CHECAR FONTE!!
[a_val, a_vet] = np.linalg.eig(Z_cov) 
idx = a_val.argsort()[::-1]
a_val1 = a_val[idx]
a_vet1 = a_vet[:,idx]

# PoV - Percentual de Variância
PoV = (100 * a_val1) / sum(a_val1)
PoV = PoV.round(4).real
PoVcum = PoV[0:6].cumsum().round(4).real

# Carga Fatorial Ar 
Ar = a_vet1 * (a_val1**(0.5))

# Plot do comportamento espacial das EOF's 
for c in range(0, 3): 
    plt.subplot(2, 2, (c+1))
    eof = np.reshape(Ar[:, c], [1, 30, 30])
    eof = np.transpose(eof).squeeze()
    plt.clabel(plt.contour(lon, lat, eof))
    plt.colormaps()
    plt.colorbar()
    plt.title(f'EOF{c+1}')
    plt.xlabel('lon')
    plt.ylabel('lat')
plt.show()

t = np.arange(1, 25, 1)
"""


t = np.arange(1, 25, 1)

# Plot das séries temporais das EOF's
for c in range(0, 3):
    plt.subplot(2, 2, (c+1))
    pcts = np.dot(Z, a_vet1,[:, c])
    plt.plot(t, pcts)
    plt.title(f'Série Temporal EOF{c+1}')
    plt.xlabel('Tempo')
    plt.ylabel('Amplitude EOF')
plt.show()
"""