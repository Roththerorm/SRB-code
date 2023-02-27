'''
This script generates the graphs for the figures. It's a simple way to organize all data graphs in just one set of functions
'''

import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : 12})
matplotlib.rcParams['axes.linewidth'] = 1.


def energy_fluctuation(*args, title, save, fig_path):

    '''
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
    }
    '''
    # Creating axes
    fig, ax = plt.subplots()

    # Plot and adjusts
    x_line = np.linspace(0, len(args[2]), len(args[2]))
    ax.scatter(x_line, args[2], s=4, c='cadetblue')
    ax.set_ylim(min(args[2]) - 100, max(args[2]) + 100)
    
    # Stripe defining points used do calculate the correlation coefficient
    ax.axhspan(args[3], args[4], color='navy', alpha=0.4, label=f'{args[5] + args[6]} points in region')


    # Text on graph
    ax.set_ylabel('Power (arb. unit)', labelpad= 15)
    ax.set_xlabel('Time (samples)', labelpad= 15)
    ax.legend(loc='lower right')

    if title == 'y':
        ax.set_title(f'Total energy per time of collection for {args[0]} mV (round: {args[1]}) \n ({args[7] * 100}% used)', pad=15)
    
    if save == 'y':
        plt.savefig(fr'{fig_path}\\power_time{args[0]}_{args[1]}.png')


def parisi_histogram(*args, title, save, fig_path):

    '''
    Here's the following order for arguments parameters
    *args = {
        [0] k : current value
        [1] l : round value
        [2] parameters : array of parisi parameters 
        [3] percentage : % used to select the interval 
    }
    '''

    fig, ax = plt.subplots()
    
    ax.hist(args[2], bins= math.floor(np.sqrt(len(args[2]))), density=True, facecolor='steelblue')

    ax.tick_params(which='both', direction="in")
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
   
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.set_xlim(-1.1,1.1)

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_ticks = ax.xaxis.get_ticklabels()
    remove_ticks = [2,4,6,8]
    for i in remove_ticks:
        x_ticks[i].set_visible(False)

    # Change the text for the Parisi histogram
    if title == 'y':
        ax.set_title(f'Parisi coefficient for {args[0]/10} W (round: {args[1]}) \n ({args[3]*100}% used)', pad=15)
    
    ax.set_xlabel('$q$', labelpad=15)
    ax.set_ylabel('$P(q)$', labelpad=15)
    ax.get_tightbbox()
    
    if save == 'y':
        plt.savefig(fr'{fig_path}\\parisi_coff_{args[0]}_{args[1]}.png')