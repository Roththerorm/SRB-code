'''
This script calculates the correlation pattern between two peaks of the spectrum
'''

import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

spectrum_values = [390,1]
data_length = 3000
noise = 700
min_wave = 0
max_wave = 400
height_vector = []

path = r'D:\LaserYb\Medidas Espectrometro\17_02_2023\binary_data'

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
    
    for r in range(1, len(data[0])):
        
        spectrum = df.iloc[:, r].values
    
        peaks, _ = find_peaks(spectrum, height = noise, distance=2, prominence=10)

        if len(peaks) > 1:
            prop = pd.DataFrame.from_dict(_)
            
            (first_max_idx, second_max_idx) = prop['prominences'].nlargest(2).index.tolist()

            max_indices = sorted([first_max_idx, second_max_idx])

            max_values = prop.iloc[max_indices]['peak_heights']

            height_vector.append(max_values.values)
        else:
            pass

final_data = pd.DataFrame(height_vector, columns=['First peak', 'Second peak'])

corr = final_data.corr(method='pearson')

corr.columns = ['First peak', 'Second peak']

# fig, ax = plt.subplots()
# ax.scatter(final_data['Peak_one'], final_data['Peak_two'], s=1)
# corr = np.corrcoef(height_vector)
ax = sns.heatmap(corr, annot=True, cmap='YlOrRd', vmin=-1, vmax=1, cbar_kws={'label': 'Pearson correlation'})
c_bar = ax.collections[0].colorbar
c_bar.set_ticks([-1, 0, 1])
plt.savefig(rf'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Escrita\Universal manuscript template for Optica Publishing Group journals\figures\heat_map.pdf')
 



