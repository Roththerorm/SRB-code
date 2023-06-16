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

# import graphs
from power_calculation import compute_power
from exclusion import remove_outliers
from fluctuations import get_max_values
from FillBetween3d import fill_between_3d
from spectrum_animation import plot_animation


from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.transforms as mtransforms
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.ticker as ticker

############# GLOBAL VARIABLES #############
              
pico = 0                # Pick taken in the samples of power
percentage = 0.5        # Filter used to select the points
num_points = 3000     # Number of points to calculate Parisi Coefficient
total_points = 5000     # Total number of replicas
first_binary_data = 298
last_binary_data = 298
step = 2
mod_q = []


i = 0
size = 14
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : size})
matplotlib.rcParams['axes.linewidth'] = 1.
matplotlib.rcParams['legend.handlelength'] = 0


############# MAIN CODE #############

path_in = r'D:\LaserYb\Medidas Espectrometro\mes 01_23\18_01_2023\binary_data'
plot_places = ['(a)', '(b)']
lable_names=['não fode']
list_plots = [49]
l = 1 
count = 0

fig, axs = plt.subplot_mosaic([['(a)', '(b)']], figsize=(10.1,4), gridspec_kw=dict(wspace=0.3))

for label, ax in axs.items():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(15/72, -10/72, fig.dpi_scale_trans)
    
    # Set x-coordinate of label depending on plot position
    xcoord = 0.0  # Left plot
    halign = 'left'
    color_font='black'

    ax.text(xcoord, 1.0, label, transform=ax.transAxes + trans,
            fontsize=size, verticalalignment='top', horizontalalignment=halign,
            fontfamily='sans-serif', color=color_font)


def parisi_histogram(*args, show_title, show_bottom, show_label, label, font_size, save, fig_path, format):

    '''
    Here's the following order for arguments parameters
    *args = {
        [0] k : current value
        [1] l : round value
        [2] parameters : array of parisi parameters 
        [3] percentage : % used to select the interval
        [4] Axis of the figures
    }
    '''

    if show_label == 'y':

        args[4].hist(args[2], bins= math.floor(np.sqrt(len(args[2]))/4), density=True, facecolor='firebrick', alpha=0.75, label=f'{label}')
        args[4].legend(loc='upper right', handlelength=0, handletextpad=0, fancybox=False, frameon=False, fontsize=font_size)

    elif show_label == 'n':
        
        args[4].hist(args[2], bins= math.floor(np.sqrt(len(args[2]))/4), density=True, facecolor='firebrick', alpha=0.75)
    
    # Change the text for the Parisi histogram
    if show_title == 'y':
        args[4].set_title(f'Parisi coefficient for {args[0]/10} W (round: {args[1]}) \n ({args[3]*100}% used)', pad=15)
    
    # Remove intermediate of the x axis. More specific it removes {-0.75, -0.25, 0.25, 0.75} from the graph
    args[4].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_ticks = args[4].xaxis.get_ticklabels()
    # remove_ticks = [2,4,6,8]
    # for i in remove_ticks:
    #     x_ticks[i].set_visible(False)
    
    if show_bottom == 0:
        # Set the axis labels
        args[4].set_xlabel('$q$', labelpad=15)
    else:
        plt.setp(args[4].get_xticklabels(), visible=False)

    
    args[4].set_ylabel('$P(q)$', labelpad=15)

    # Configuration of ticks (major and minor)
    args[4].tick_params(axis='both', which='both', direction="in", labelsize=size)
    args[4].tick_params(which='both', bottom=True, left=True)
    args[4].xaxis.set_minor_locator(AutoMinorLocator())
    args[4].yaxis.set_minor_locator(AutoMinorLocator())
    
    # Set limit and configure legend
    args[4].set_xlim(-1.1,1.1)
    
    if save == 'y':
        plt.tight_layout()
        plt.savefig(fr'{fig_path}\\parisi_coff_{args[0]}_{args[1]}.{format}')
    
    else:
        pass

def spectrum_plots(*args):

    spectrum_values = [[list_plots[0],1]]
    colors = ['firebrick']

    for s in range(len(spectrum_values)):

        path = path_in
        filename = f'b_data_{spectrum_values[s][0]}_{spectrum_values[s][1]}.npy'
        filepath = os.path.join(path, filename)

        if not os.path.exists(filepath):
            print(f'Error: File "{filepath}" does not exist')
        else:
            with open(filepath, 'rb') as f:
                data = np.load(f)


            # Use list comprehension to generate the list of column names
            columns = ['Wavelengths'] + [f'Intensity_{i}' for i in range(1, len(data[0]))]

            # Use f-strings to format the column names
            f_data = pd.DataFrame(data, columns=columns)
            f_data['Wavelengths'] = f_data['Wavelengths'] + 2.5

            max_value = f_data.iloc[:, 1824].max()
            min_value = f_data.iloc[:, 1824].min()
            dif = max_value - min_value
            args[0].plot(f_data.iloc[:69, 0], f_data.iloc[:69, 2].subtract(min_value).div(dif), lw=1.5, c=colors[s])
            args[0].tick_params(axis='both', which='major', labelsize=size)
            args[0].set_xlabel('Wavelength (nm)', labelpad=15)
            args[0].set_ylabel('Normalized Intensity', labelpad=15)
            args[0].minorticks_on()
            args[0].tick_params(axis='both', which='both', direction='in')


for k in list_plots:

    r = 0
    
    print(f'Current file ({k}) ({((k-first_binary_data)//step) + 1} of {int((last_binary_data - first_binary_data)/2)}). Round: {l}')
    
    path = path_in
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
        f_data['Wavelengths'] = f_data['Wavelengths'] + 2.5
        
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

        pearson_dataframe = f_data.iloc[110:238,1:].T

        # Get the average intensity I for each k
        f_data['sum'] = f_data.iloc[:,1:].sum(axis=1) / outrange

        # Remove the 'Wavelengths' and 'Sum' columns
        f_data_subset = f_data.iloc[:, 1:-1].subtract(f_data['sum'], axis=0)

        # Retrieve the name of the columns
        columns = f_data.columns

        # Calculate the Parisi Coefficient 
        correlation = f_data_subset.corr(method='pearson')

        pearson_correlation = pearson_dataframe.corr(method='pearson')

        labels_values = (f_data.iloc[110:238,0].values)
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

        # ##################### PLOTS #################################
        
            
        args_2 = (k, l, parameters, percentage, axs['(b)'])

        parisi_histogram(*args_2, show_title='n', show_bottom=0, label=lable_names[count], show_label='n', font_size=size, save='n', fig_path=figure_path, format='pdf')

        spectrum_plots(axs['(a)'])


# with open (rf'D:\LaserYb\Medidas Espectrometro\17_02_2023\max.txt', 'w') as f:
#     for q in mod_q:
#         f.write(f'{q}\n')

# fig.subplots_adjust(wspace=25)
plt.savefig(r'C:\Users\nicol\Desktop\Figuras (Marcio)\SM4.pdf', bbox_inches='tight')
#plt.show()-