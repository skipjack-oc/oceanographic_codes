"""
@author: skipjack - Vladmyr Schlosser Mello
Adapted from: Vinicius Dionysio Alves (https://github.com/ocdionysio)

Fast Fourier Transform (FFT) analisys from tide data.

data source (series.mat): hypothetical dataset 
data source: http://www.cambridge.org/glover
"""
# Load libraries
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hypothetical data

# Load .mat files
f = sio.loadmat('series.mat')

# Construct dataframe
df = pd.DataFrame(columns = ['t','y'])
df['t'] = f['t'].T.squeeze()
df['y'] = f['y'].T.squeeze()

# Set sample size
N = len(df['y'])

y = df['y'].copy()
x = np.arange(0,N)  # Vector time

# Time serie plot
plt.plot(x, y)
plt.xlim(x[0],x[N-1])
plt.title('Time serie - Random data')
plt.xlabel('Time')
plt.ylabel('Amplitude (m) ')
plt.show()

# Find closest square for FFT
n = int(2**np.ceil(np.log2(N)))

# Remove mean
y = y[:] - df['y'].mean()

# FFT
Y = np.fft.fft(y, n)

# Find power
Pyy = (Y.conjugate() * Y)/n
Pyy = Pyy.real

# Frequency quantity after 0
m = n//2
deltaT = (N-1)/N
fc = 1/(2*deltaT) #  critical frequency
f = fc * np.arange(0,m+1)/m # frequency vector

Pyy = Pyy[:m+1]
Pyy[1:] = 2 * Pyy[1:]

# Find peaks and their indexes
sort = Pyy.copy()
sort[::-1].sort()
sort = pd.Series(sort)
i = np.zeros(5)

for c in range(0, 5):
    i[c] = np.where(Pyy==sort[c])[0]
i = np.intp(i)

# Highest powers and their frequencies print
print(sort[:5])
print(pd.Series(f[i]))

# Periodogram plot, 5 major power peaks
string = ['1','2','3','4','5']
plt.loglog(f[1:], Pyy[1:], color='gray')
for c in range(0,5):
    plt.annotate(string[c],xy=[f[i[c]],Pyy[i[c]]],arrowprops=dict(arrowstyle = "->",connectionstyle = "angle,angleA=45,angleB=0,rad=10"))
plt.title('Periodogram - Random serie')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.show() 

##########################################################################################################
# Real data

# Load .dat files and fit for dataframe
col = ['ID', 'year', 'month', 'day', 'hour', 'minute', 'pred', 'ver']
f = np.loadtxt('woods_hole_tides_2005_05.dat')
data = pd.DataFrame(f, columns=col)

# Fit datetime column
date = pd.DataFrame(data[col[1:6]])
date = pd.to_datetime(date, format='%Y-%m-%D %H:%M')
data.insert(8, 'date', date.map(lambda x: x.strftime('%d %H:%M')))

# FFT
N = len(data['ver'])
n = int(2**np.ceil(np.log2(N)))
y = data['ver'][:] - data['ver'].mean()

Y = np.fft.fft(y, n)
Pyy = (Y.conjugate() * Y)/n # power

m = n//2
deltaT = (31*24)/N
fc = 1/(2*deltaT) #  critical frequencie
f = fc * np.arange(0,m+1)/m #  frequencies for plotting

Pyy = Pyy[:m+1]
Pyy[1:] = 2 * Pyy[1:]

# Sort powers and find 8 power peaks
sort = Pyy.copy()
sort[::-1].sort()
sort = pd.Series(sort)
i = np.zeros(8)

for c in range(0, 8):
    i[c] = np.where(Pyy==sort[c])[0]
i = np.intp(i)

# Periodogram plot for 8 power peaks
string = ['1','2','3','4','5','6','7','8']

plt.loglog(f[1:], Pyy[1:], color='gray')
for c in range(0,8):
    plt.annotate(string[c],xy=[f[i[c]],Pyy[i[c]]],arrowprops=dict(arrowstyle = "->",connectionstyle = "angle,angleA=45,angleB=0,rad=10"))
plt.title('Periodogram - Tide Data Woods Hole (2005/05)')
plt.xlabel('Frequency (h-¹)(log)')
plt.ylabel('Power(m²)(log)')
plt.show()

# Peak period print
p = pd.Series(1/f[i])
print(p)

# Parseval theorem test and round 4 decimal
if(np.round(sum(abs(y)**2),4)==np.round(sum(Pyy),4)):
    print('Parseval theorem is valid')
   
