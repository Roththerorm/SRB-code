
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import matplotlib
import os
import numpy as np

size = 12

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : size})
matplotlib.rcParams['axes.linewidth'] = 1.

def q_max():

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

        fig, ax = plt.subplots()
        ax.plot(watt, content_list, '-',lw=1.5, c='firebrick', zorder=1, alpha=0.75)
        ax.scatter(watt, content_list, s=30, facecolors='white', edgecolors='firebrick', zorder=3, alpha=0.75)
        ax.tick_params(axis='both', which='major', labelsize=size)
        ax.set_xlabel('Power (mW)', labelpad=15)
        ax.set_ylabel(r'$|q|$', labelpad=15)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', direction='in')
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


q_max()