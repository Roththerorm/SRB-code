import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.signal import find_peaks
import matplotlib

spectrum_values = [48,1]
data_length = 5000
noise = 700
min_wave = 0
max_wave = 500
height_vector = []
i = 1552
size = 10

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : 10})
matplotlib.rcParams['axes.linewidth'] = 1.

path = r'D:\LaserYb\Medidas Espectrometro\20_01_2023\binary_data'

filename = f'b_data_{spectrum_values[0]}_{spectrum_values[1]}.npy'

filepath = os.path.join(path, filename)

if not os.path.exists(filepath):
    print(f'Error: File "{filepath}" does not exist')
else:
    with open(filepath, 'rb') as f:
        data = np.load(f)


    # Use list comprehension to generate the list of column names
    columns = ['Wavelengths'] + [f'Intensity_{i}' for i in range(1, len(data[0]))]

    # Use f-strings to format the column names
    df = pd.DataFrame(data, columns=columns)


    maximum = df.iloc[:, i].max()
    dif = df.iloc[:, i].max() - df.iloc[:, i].min()
    
    
    fig, ax = plt.subplots()

    spectrum = df.iloc[:, i].values
    
    peaks, _ = find_peaks(spectrum, height = noise, distance=2, prominence=10)
    
    peak_values = spectrum[peaks]

    if peak_values is not None and len(peaks) > 10:
        
        prominences = _['prominences']

        peak_heights = _['peak_heights']
        
        # sort the peaks by their prominence values
        sorted_peaks = sorted(zip(peaks, prominences, peak_heights), key=lambda x: x[1], reverse=True)

        # select the two peaks with the highest prominences
        top_two_peaks = [sorted_peaks[0][0], sorted_peaks[1][0]]

        top_two_peak_heights = [sorted_peaks[0][2], sorted_peaks[1][2]]

        ax.plot(df.iloc[min_wave:max_wave, 0], df.iloc[min_wave:max_wave, i].subtract(maximum).div(dif) + 1, lw=1.5, color='firebrick')
        ax.tick_params(axis='both', which='major', labelsize=size)
        ax.plot(df.iloc[top_two_peaks,0], df.iloc[top_two_peaks,i].subtract(maximum).div(dif) + 1, 'v', color='black', label='Prominent peaks')
        ax.set_xlabel('Wavelength (nm)', labelpad=15)
        ax.set_ylabel('Normalized intensity', labelpad=15)
        ax.tick_params(axis='both', direction='in')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    
    else:
        
        ax.plot(df.iloc[min_wave:max_wave, 0], df.iloc[min_wave:max_wave, i].subtract(maximum).div(dif) + 1, lw=1.5, color='firebrick', label = 'P\u209A\u1D64\u2098\u209A = 4.8 W')
        ax.tick_params(axis='both', which='major', labelsize=size)

        ax.set_xlabel('Wavelength (nm)', labelpad=15)
        ax.set_ylabel('Normalized intensity', labelpad=15)
        ax.tick_params(axis='both', direction='in')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(rf'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Escrita\Universal manuscript template for Optica Publishing Group journals\figures\mode_locked_TiSa.pdf')