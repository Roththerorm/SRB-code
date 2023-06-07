import os
import math
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter


pico = 0                # Pick taken in the samples of power
percentage = 1        # Filter used to select the points
num_points = 3000      # Number of points to calculate Parisi Coefficient
total_points = 3000     # Total number of replicas
first_binary_data = 240
last_binary_data = 242
step = 2
std_deviations = []

size = 12
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : 12})
matplotlib.rcParams['axes.linewidth'] = 1.


full_data = pd.DataFrame()
#340, 344, 348, 240
# index=[i for i in range(1000, 12200,1)]

for k in range( first_binary_data, last_binary_data, step):
    for l in range(1,2):

        if k == 340 or k == 344 or k == 348:
            pass
        else:

            print(f'Current file ({k}) ({((k-first_binary_data)//step) + 1} of {int((last_binary_data - first_binary_data)/2)}). Round: {l}')
            
            path = r'D:\LaserYb\Medidas Espectrometro\mes 02_23\17_02_2023\binary_data'
            figure_path = r'D:\LaserYb\Resultado de Tratamento de Dados\Maxima_analysis\17_02_23'

            filename = f'b_data_{k}_{l}.npy'
            filepath = os.path.join(path, filename)

            # Use os.path.exists to check if the file exists
            if not os.path.exists(filepath):
                print(f'Error: File "{filepath}" does not exist')
            else:
                # Use the with statement to open the file
                with open(filepath, 'rb') as f:
                    data = np.load(f)

                # print('> Loading data frame')
                # Use list comprehension to generate the list of column names
                columns = ['Wavelengths'] + [f'Intensity_{i}' for i in range(1, len(data[0]))]

                # Use f-strings to format the column names
                f_data = pd.DataFrame(data, columns=columns)

                max_values = f_data.iloc[:,1:].max()

                max_values_normalized = max_values.subtract(min(max_values)).div(max(max_values)-min(max_values))

                hist, bin_edges = np.histogram(max_values_normalized, bins=math.floor(np.sqrt(len(max_values))), density=False)

                hist_normalized = [(hist[i]-min(hist))/(max(hist)-min(hist)) for i in range(len(hist))]

                bin_edges = [round(bin_edges[i],3) for i in range(len(bin_edges)-1)]

                temp_dataframe = pd.DataFrame(hist_normalized, index=bin_edges, columns=[f'data_{k}'])

                full_data = full_data.join(temp_dataframe, how='outer')


#             std_dev = max_values.std()
#             std_deviations.append(std_dev)

#             fig, ax = plt.subplots()
            
            # ax.hist(max_values, bins= math.floor(np.sqrt(len(max_values))), density=True, facecolor='cadetblue', alpha=0.75)
            # ax.set_title(f'Maxima distribution for {k} mV (round: {l})\n$\sigma = {round(std_dev,2)}$', pad=15)
            # ax.set_xlabel('$n$', labelpad=15)
            # ax.set_ylabel('$P(n)$', labelpad=15)
            # ax.tick_params(axis='both', which='both', direction="in", labelsize=size)
            # ax.tick_params(which='both', bottom=True, left=True)
            # ax.xaxis.set_minor_locator(AutoMinorLocator())
            # ax.yaxis.set_minor_locator(AutoMinorLocator())
            # plt.tight_layout()
            # plt.savefig(fr'{figure_path}\\maxima_dist_{k}_{l}.png')



full_data = full_data.fillna(0).T
full_data_smooth = gaussian_filter(full_data, sigma=1)
sns.heatmap(full_data_smooth, annot=False, cmap='plasma')
plt.tight_layout()
plt.show()