# Imported full libraries
import os
import math
import random
import cycler
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Imported modules from libraries and Python codes
from cProfile import label
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from functools import partial

import graphs
from power_calculation import compute_power
from exclusion import remove_outliers
from fluctuations import get_max_values
from FillBetween3d import fill_between_3d
from spectrum_animation import plot_animation


from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.ticker as ticker

############# GLOBAL VARIABLES #############
              
pico = 0                # Pick taken in the samples of power
percentage = 0.5        # Filter used to select the points
num_points = 3000      # Number of points to calculate Parisi Coefficient
total_points = 5000     # Total number of replicas
first_binary_data = 298
last_binary_data = 298
step = 2
mod_q = []


i = 0
colors = ['#347B98', '#E12514', '#5EBA1C']

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : 12})
matplotlib.rcParams['axes.linewidth'] = 1.
matplotlib.rcParams['legend.handlelength'] = 0


############# MAIN CODE #############

list_plots = [48]
l = 1 
#, 290, 214
for k in list_plots:

    r = 0
    
    print(f'Current file ({k}) ({((k-first_binary_data)//step) + 1} of {int((last_binary_data - first_binary_data)/2)}). Round: {l}')
    
    path = r'D:\LaserYb\Medidas Espectrometro\20_01_2023\binary_data'

    figure_path = r'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Escrita\Universal manuscript template for Optica Publishing Group journals\figures'

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
        
        print('> Selecting interval range')

        # plot_animation(f_data, 100, 250, 0, 11000, 700, 'y', 10, False)

        # Computes the total power of each replica
        power_samples = compute_power(f_data, data, trapezoid)

        # Filters out the data only to have the amount of points required
        f_data, outrange, disc, lim_max, lim_min = remove_outliers(power_samples, percentage, pico, num_points, total_points, f_data)

        # Shows how many points were discarded. The number shows representes the points taken inside the percentage
        # minus the total points wanted
        print(f'Were discarded {disc} from {total_points}')
        print('> Calculating Parisi coefficients...')

        pearson_dataframe = f_data.iloc[:,1:].T

        # Get the average intensity I for each k
        f_data['sum'] = f_data.iloc[:,1:].sum(axis=1) / outrange

        # Remove the 'Wavelengths' and 'Sum' columns
        f_data_subset = f_data.iloc[:, 1:-1].subtract(f_data['sum'], axis=0)

        # Retrieve the name of the columns
        columns = f_data.columns

        # Calculate the Parisi Coefficient 
        correlation = f_data_subset.corr(method='pearson')

        pearson_correlation = pearson_dataframe.corr(method='pearson')

        labels_values = (f_data.iloc[:,0].values)
        labels = [int(i) for i in labels_values]
        pearson_correlation.columns = labels
        pearson_correlation.index = labels

        print('Excluding duplicates')

        # Flatten the array and remove the main diagonal elements
        flat_array = correlation.values.flatten()
        flat_array = flat_array[np.arange(flat_array.size) % (correlation.shape[0] + 1) != 0]

        # Create a list of the unique, non-NaN elements
        parameters = list(set(flat_array))

        print('Done!')
        print('> Final calculations and preparing plots')

        # hist, bin_edges = np.histogram(parameters, density=True, bins=math.floor(np.sqrt(len(parameters))))

        # max_q = abs(bin_edges[np.argmax(hist)])

        # mod_q.append(max_q)
        
        # ##################### PLOTS #################################
        
        # args_1 = (k, l, power_samples, lim_min, lim_max, num_points, disc, percentage)


    #     args_2 = (k, l, parameters, percentage, ax1)
    
    # # graphs.energy_fluctuation(*args_1, title='n', save='n', fig_path=figure_path)


            
        fig, ax = plt.subplots(num=1)

        args_2 = (k, l, parameters, percentage, ax)

        graphs.parisi_histogram(*args_2, show_title='n', show_label='n', save='y', fig_path=figure_path, format='pdf')
        
        # fig, ax2 = plt.subplots()
        # heatmap = sns.heatmap(pearson_correlation, vmin=-1, vmax=1, cmap='YlOrRd', xticklabels=100, yticklabels=100, cbar_kws={'label': 'Pearson correlation'}, ax=ax2)
        
        # heatmap.tick_params(which='both', direction="in")
        # plt.setp(heatmap.get_xticklabels(), visible=False)
        # heatmap.set_xlabel('Wavelength (nm)', labelpad=15)
        # heatmap.set_ylabel('Wavelength (nm)', labelpad=15)
        # plt.tight_layout()
        # plt.savefig(r'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Escrita\Universal manuscript template for Optica Publishing Group journals\figures\pearson_a_mode.pdf')
  

# with open (rf'D:\LaserYb\Medidas Espectrometro\17_02_2023\max.txt', 'w') as f:
#     for q in mod_q:
#         f.write(f'{q}\n')




