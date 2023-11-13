import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib
import statistics

size = 12
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : size})
matplotlib.rcParams['axes.linewidth'] = 1.


std_data = pd.read_csv('sigma_array_3.csv').values.flatten().tolist()

std_copy = std_data.copy()

pulse_train_data = pd.read_csv('29fse_train30.csv', sep=',', header=0)


pulse_train_data['second'] *= 1e9
pulse_train_data['Volt'] *= 1e3

x = np.linspace(0, 2373.9390 / 60, len(std_data))

linear_fit = np.poly1d(np.polyfit(x,std_data,1))

points = list(reversed(linear_fit(x)))
new_points = std_data

for i in range(len(std_data)):
    diff = abs(new_points[i] - points[i])
    if new_points[i] >= points[i]:
        std_data[i] -= diff/2
    else:
        std_data[i] += diff/2

mean = statistics.mean(std_data)
std = statistics.stdev(std_data)
print(mean, std)



fig, ax = plt.subplots()
ax.scatter(x, std_data, s=0.5, c='crimson')
ax.tick_params(axis='both', which='both', direction="in", labelsize=size)
ax.plot([], [], '', label=r'$\tau = 275.5$ fs')
ax.plot([], [], '', label=r'$\sigma = 0.5$ fs')
ax.set_xlabel('time (min)', labelpad=15)
ax.set_ylabel('Adjusted FWHM (fs)', labelpad=15)
ax.set_ylim(265, 290)
ax.set_xlim(0, max(x))
ax.legend(alignment='right')
plt.tight_layout()
plt.legend()


fig, ax2 = plt.subplots()
ax2.plot(pulse_train_data.iloc[:,0], pulse_train_data.iloc[:,1], lw=1, c='crimson', label='Pulse Train')
ax2.tick_params(axis='both', which='both', direction="in", labelsize=size)
ax2.set_xlabel('time (ns)', labelpad=15)
ax2.set_ylabel('Amplitude (mV)', labelpad=15)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
plt.tight_layout()
plt.legend()

fig, ax3 = plt.subplots()
ax3.scatter(x, std_copy, s=0.5, c='crimson', label='Sample')
#ax3.plot(x, linear_fit(x), lw=1.5, c='black', label="Linear fit: $-0.2613 x + 280.7$")
ax3.tick_params(axis='both', which='both', direction="in", labelsize=size)
ax3.set_xlabel('time (min)', labelpad=15)
ax3.set_ylabel('FWHM (fs)', labelpad=15)
ax3.set_ylim(260, 290)
ax3.legend(alignment='center')

plt.tight_layout()

plt.show()


