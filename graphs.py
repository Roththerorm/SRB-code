'''
This script generates the graphs for the figures. It's a simple way to organize all data graphs in just one set of functions.

Created by: @Nicolas Pessoa
'''

import os
import math
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter

size = 12

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : size})
matplotlib.rcParams['axes.linewidth'] = 1.


def power_fluctuation(*args, show_title, save, fig_path, format):

    '''
    This function produces a graph of the power fluctuations of all replicas for a specific current.
    Here's the following order for arguments parameters
    *args = {
        [0] k : current value
        [1] l : round value
        [2] power_samples : list of power calculated for each replica 
        [3] lim_min : minimum value from average peak of the energy histogram
        [4] lim_max : maximum value from average peak of the energy histogram
        [5] num_points : total points collected
        [6] disc: numbers of points discarded 
        [7] percentage : % used to select the interval
        [8] axes: axes of figures
    }
    show_title: if 'y', show the title in the plot;
    
    save: if 'y' saves the figure in the path given;
    
    format: insert a valid file type to save the archive;
    '''

    # Plot and adjusts
    x_line = np.linspace(0, len(args[2]), len(args[2]))
    args[8].scatter(x_line, args[2], s=4, c='cadetblue')
    args[8].set_ylim(min(args[2]) - 100, max(args[2]) + 100)
    
    # Stripe defining points used do calculate the correlation coefficient
    args[8].axhspan(args[3], args[4], color='navy', alpha=0.4, label=f'{args[5] + args[6]} points in region')

    # Text on graph
    args[8].set_ylabel('Power (arb. unit)', labelpad= 15)
    args[8].set_xlabel('Time (samples)', labelpad= 15)
    args[8].tick_params(axis='both', which='major', labelsize=size)
    args[8].legend(loc='lower right')

    if show_title == 'y':
        args[8].set_title(f'Total energy per time of collection for {args[0]} mV (round: {args[1]}) \n ({args[7] * 100}% used)', pad=15)
       
    if save == 'y':
        plt.savefig(fr'{fig_path}\\power_time{args[0]}_{args[1]}.{format}')
    
    else:
        plt.show()
    


def parisi_histogram(*args, show_title, show_label, save, fig_path, format):

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

        args[4].hist(args[2], bins= math.floor(np.sqrt(len(args[2]))/4), density=True, facecolor='firebrick', alpha=0.75, label=f'P = {round(args[0]/400,2)} P\u209C\u2095')
        args[4].legend(loc='upper right', fontsize='medium', frameon=True, handletextpad=0.1)

    elif show_label == 'n':
        
        args[4].hist(args[2], bins= math.floor(np.sqrt(len(args[2]))/4), density=True, facecolor='firebrick', alpha=0.75)
    
    # Change the text for the Parisi histogram
    if show_title == 'y':
        args[4].set_title(f'Parisi coefficient for {args[0]/10} W (round: {args[1]}) \n ({args[3]*100}% used)', pad=15)
    
    # Remove intermediate of the x axis. More specific it removes {-0.75, -0.25, 0.25, 0.75} from the graph
    args[4].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_ticks = args[4].xaxis.get_ticklabels()
    remove_ticks = [2,4,6,8]
    for i in remove_ticks:
        x_ticks[i].set_visible(False)
    
    # Set the axis labels
    args[4].set_xlabel('$q$', labelpad=15)
    # plt.setp(args[4].get_xticklabels(), visible=False)
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
        plt.show()



def q_max(*args):

    '''
    Here's the following orders for arguments parameters
    *args = {
    [0] - Axis
    }
    '''
    def current_to_watt(num):

        return round((0.7412 * (num) - 15.276), 3) 
    
    with open(r'D:\LaserYb\Medidas Espectrometro\17_02_2023\max1.txt') as data:
        content = data.read()
        content_list = list(map(float, content[:-1].split('\n')))
        
        current = [(410 - x * 0.8) for x in range(0, 214, 2)]
        watt = list(map(current_to_watt,current))
        watt.reverse()
        indexes = [75,73,71,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
        for i in indexes:
            del content_list[i]
            del watt[i]

        args[0].plot(watt, content_list, '-',lw=1.5, c='firebrick', zorder=1, alpha=0.75)
        args[0].scatter(watt, content_list, s=30, facecolors='white', edgecolors='firebrick', zorder=3, alpha=0.75)
        args[0].tick_params(axis='both', which='major', labelsize=size)
        args[0].set_xlabel('Power (mW)', labelpad=15)
        args[0].set_ylabel(r'$|q_{max}|$', labelpad=15)
        args[0].minorticks_on()
        args[0].tick_params(axis='both', which='both', direction='in')
        plt.tight_layout()
        plt.savefig(rf'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Escrita\Universal manuscript template for Optica Publishing Group journals\figures\q_max_adjusted.pdf')


def spectrum_plots():

    spectrum_values = [[408,1],[320,1],[244,1]]
    power = ['287 mW', '234 mW','189 mW']
    colors = ['#5EBA1C', '#E12514', '#347B98']

    fig, ax = plt.subplots()

    for s in range(len(spectrum_values)):

        path = r'D:\LaserYb\Medidas Espectrometro\17_02_2023\binary_data'
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

            max_value = f_data.iloc[:, 1824].max()
            min_value = f_data.iloc[:, 1824].min()
            dif = max_value - min_value
            ax.plot(f_data.iloc[:, 0], f_data.iloc[:, 1824], lw=1.5, label=f'{power[s]}', c=colors[s])
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlabel('Wavelength (nm)', labelpad=15)
            ax.set_ylabel('Intensity (arb. units)', labelpad=15)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in')
            ax.legend(loc='best')

    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(rf'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Escrita\Universal manuscript template for Optica Publishing Group journals\figures\spectrum_not_normalized.pdf')