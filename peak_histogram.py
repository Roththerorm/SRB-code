import os
import math
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator
import matplotlib.transforms as mtransforms



pico = 0                # Pick taken in the samples of power
percentage = 1        # Filter used to select the points
num_points = 3000      # Number of points to calculate Parisi Coefficient
total_points = 3000     # Total number of replicas
first_binary_data = 198
last_binary_data = 412
step = 2
std_deviations = []
peaks_index = [174, 179]

size = 16.5
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : 16.5})
matplotlib.rcParams['axes.linewidth'] = 1




#340, 344, 348, 240
#['(a)', '(a)','(b)', '(b)','(c)', '(c)'], ['(d)', '(d)','(d)', '(e)','(e)', '(e)']

fig, axs = plt.subplot_mosaic([['(a)', '(c)'], ['(a)', '(c)'], ['(a)', '(d)'], ['(b)', '(d)'], ['(b)', '(e)'], ['(b)', '(e)']],
                              figsize=(11,8), gridspec_kw = {'hspace':0.25, 'wspace':0.4})


for label, ax in axs.items():
    # label physical distance in and down:
    trans = mtransforms.ScaledTranslation(8/72, -8/72, fig.dpi_scale_trans)
    
    # Set x-coordinate of label depending on plot position
    if '(c)' <= label <= '(e)': 
        trans = mtransforms.ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans, fontsize=size, verticalalignment='top', fontfamily='sans-serif', color='white')   
    else:
        xcoord = 0.00  # Left plot
        ycoord = 1.0
        halign = 'left' 
        ax.text(xcoord, ycoord, label, transform=ax.transAxes + trans,
            fontsize=size, verticalalignment='top', horizontalalignment=halign,
            fontfamily='sans-serif')

def current_to_watt(num):
    return math.floor(round((0.7412 * (num) - 15.276), 2))

current = [math.floor(410 - x * 0.8) for x in range(0, 214, 2)]
#watt = list(map(current_to_watt,current))
#watt.reverse()
current.reverse()
indexes = [80,77,75,73,71,70,58,45,44,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
for i in indexes:
    del current[i]



superposition_data = pd.DataFrame()
#in_path = r'D:\LaserYb\Resultado de Tratamento de Dados\Peak_analysis\17_02_23'

for peak in peaks_index:
    count = 0
    skip = 0
    count_2 = 0
    full_data = pd.DataFrame()
    # Path(rf'{in_path}\index_{peak}').mkdir(parents=True, exist_ok=True)
    for k in range( first_binary_data, last_binary_data, step):
        for l in range(1,2):

            if skip in indexes:
                continue
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

                    peak_selection = f_data.iloc[peak,1:]
                    max_values = peak_selection
                    
                    max_values_normalized = peak_selection.subtract(min(peak_selection)).div(max(peak_selection)-min(peak_selection))
                    
                    std_dev = max_values.std()
                    std_deviations.append(std_dev)
                    # fig, ax = plt.subplots()

                    if k == 372 and peak == 174:
                        
                        hist, bin_edges = np.histogram(max_values_normalized, bins=2*math.floor(np.sqrt(len(max_values_normalized))), density=False)

                        hist_normalized = [(hist[i]-min(hist))/(max(hist)-min(hist)) for i in range(len(hist))]

                        bin_edges = [round(bin_edges[i],5) for i in range(len(bin_edges)-1)]

                        axs['(a)'].bar(bin_edges[:], hist_normalized, width=(bin_edges[1]-bin_edges[0]), color='firebrick', alpha=0.75, label=f'$\lambda_{1}$ = 1024.4 nm')
                        
                        # axs['(d)'].hist(max_values_normalized, bins= 2 * math.floor(np.sqrt(len(peak_selection))), density=False, facecolor='firebrick', alpha=0.75, align='right', label=f'$\lambda$ = 1021.92 nm at {watt[count_2]} mW')
                        # axs['(d)'].set_title(f'Peak fluctuation for {k} mV at $\\lambda={f_data.iloc[peak, 0]}$ m \n(round: {l}) $\\sigma = {round(std_dev,2)}$', pad=15)
                        axs['(a)'].set_ylabel('Normd. Counts', labelpad=15)
                        axs['(a)'].tick_params(axis='both', which='both', direction="in", labelsize=size)
                        axs['(a)'].tick_params(which='both', bottom=True, left=True)
                        axs['(a)'].xaxis.set_minor_locator(AutoMinorLocator())
                        axs['(a)'].yaxis.set_minor_locator(AutoMinorLocator())
                        axs['(a)'].legend(loc='upper center', handlelength=0, handletextpad=0, fancybox=False, frameon=False, fontsize=size)
                        plt.setp(axs['(a)'].get_xticklabels(), visible=False)

                    if k == 372 and peak == 179:

                        hist, bin_edges = np.histogram(max_values_normalized, bins=2*math.floor(np.sqrt(len(max_values_normalized))), density=False)
                        hist_normalized = [(hist[i]-min(hist))/(max(hist)-min(hist)) for i in range(len(hist))]
                        bin_edges = [round(bin_edges[i],5) for i in range(len(bin_edges)-1)]
                        axs['(b)'].bar(bin_edges[:], hist_normalized, width=(bin_edges[1]-bin_edges[0]), color='firebrick', alpha=0.75, label=f'$\lambda_{2}$ = 1025.6 nm')

                        #axs['(e)'].hist(max_values_normalized, bins= 2 * math.floor(np.sqrt(len(peak_selection))), density=False, facecolor='firebrick', alpha=0.75, align='right', label=f'$\lambda$ = 1023.11 nm at {watt[count_2]} mW')
                        # axs['(d)'].set_title(f'Peak fluctuation for {k} mV at $\\lambda={f_data.iloc[peak, 0]}$ m \n(round: {l}) $\\sigma = {round(std_dev,2)}$', pad=15)

                        axs['(b)'].set_xlabel('Normd. Intensity', labelpad=15)
                        axs['(b)'].set_ylabel('Normd. Counts', labelpad=15)
                        axs['(b)'].tick_params(axis='both', which='both', direction="in", labelsize=size)
                        axs['(b)'].tick_params(which='both', bottom=True, left=True)
                        axs['(b)'].xaxis.set_minor_locator(AutoMinorLocator())
                        axs['(b)'].yaxis.set_minor_locator(AutoMinorLocator())
                        axs['(b)'].legend(loc='upper center', handlelength=0, handletextpad=0, fancybox=False, frameon=False, fontsize=size)


                    hist, bin_edges = np.histogram(max_values_normalized, bins=math.floor(np.sqrt(len(max_values_normalized))), density=False)

                    hist_normalized = [(hist[i]-min(hist))/(max(hist)-min(hist)) for i in range(len(hist))]

                    bin_edges = [round(bin_edges[i],2) for i in range(len(bin_edges)-1)]

                    temp_dataframe = pd.DataFrame(hist_normalized, index=bin_edges, columns=[f'{current[count]}'])

                    full_data = full_data.join(temp_dataframe, how='outer')

                    del temp_dataframe

                    count +=1
                    count_2 += 1
        skip +=1


    full_data = full_data.fillna(0)
    full_data.loc[1.0] = 0
    full_data.rename(index={0.41:0.4, 0.11:0.1},inplace=True)
    full_data = full_data.reindex(index=full_data.index[::-1])

    y_skip = 16
    # full_data_smooth = gaussian_filter(full_data, sigma=1)
    # sns.heatmap(full_data, annot=False, cmap='magma')
    if peak == 174:
        superposition_data = full_data * 0
        superposition_data = superposition_data.add(full_data, fill_value=0)
        heatmap = sns.heatmap(full_data, vmin=0, vmax=1, cmap='magma', annot=False, xticklabels=10, yticklabels=y_skip, ax=axs['(c)'], cbar_kws={'label': 'Normd. Counts', 'ticks':[0.0,0.5,1.0], 'shrink': 0.85})
        plt.setp(heatmap.get_xticklabels(), visible=False)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, va='center')
        heatmap.set_ylabel('Normd. Intensity', labelpad=15)
    elif peak == 179:
        superposition_data = superposition_data.add(full_data, fill_value=0).div(2)
        heatmap = sns.heatmap(full_data, vmin=0, vmax=1, cmap='magma', annot=False, xticklabels=10,  yticklabels=y_skip, ax=axs['(d)'], cbar_kws={'label': 'Normd. Counts', 'ticks':[0.0,0.5,1.0], 'shrink': 0.85})
        heatmap.set_ylabel('Normd. Intensity', labelpad=15)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, va='center')
        plt.setp(heatmap.get_xticklabels(), visible=False)

        heatmap_2 = sns.heatmap(superposition_data, vmin=0, vmax=1, cmap='magma', annot=False, xticklabels=10, yticklabels=y_skip, ax=axs['(e)'], cbar_kws={'label': 'Normd. Counts', 'ticks':[0.0,0.5,1.0], 'shrink': 0.85})
        heatmap_2.set_yticklabels(heatmap_2.get_yticklabels(), rotation=0, va='center')
        heatmap_2.set_xlabel('Current (mA)', labelpad=15)
        heatmap_2.set_ylabel('Normd. Intensity', labelpad=15)

values = ['(c)', '(d)', '(e)']
for k in values:
    axs[k].axvline(x=13, ymin=0, ymax=1, color='whitesmoke', lw=2.5, ls='--')
    axs[k].axvline(x=70, ymin=0, ymax=1, color='whitesmoke', lw=2.5, ls='--')

axs['(d)'].annotate('', xy=(.5325, .915), xytext=(.5825, .915), xycoords='figure fraction', textcoords='figure fraction', arrowprops=dict(lw=1.6, arrowstyle="<->"))

axs['(d)'].annotate('', xy=(.5745, .915), xytext=(.7775, .915), xycoords='figure fraction', textcoords='figure fraction', arrowprops=dict(lw=1.6, arrowstyle="<->"))

axs['(d)'].annotate('', xy=(.7685, .915), xytext=(.802, .915), xycoords='figure fraction', textcoords='figure fraction', arrowprops=dict(lw=1.6, arrowstyle="<->"))

axs['(d)'].annotate('CW', xy=(.559, .935), xycoords='figure fraction', textcoords='figure fraction', ha='center', va='center', fontfamily='sans-serif', fontsize=size-2)

axs['(d)'].annotate('QML', xy=(.676, .935), xycoords='figure fraction', textcoords='figure fraction', ha='center', va='center', fontfamily='sans-serif', fontsize=size-2)

axs['(d)'].annotate('SML', xy=(.785, .935), xycoords='figure fraction', textcoords='figure fraction', ha='center', va='center', fontfamily='sans-serif', fontsize=size-2)

plt.savefig(r'C:\Users\nicol\Desktop\Figuras (Marcio)\fig5.pdf', bbox_inches='tight')


# def current_to_watt(num):

#     return round((0.7412 * (num) - 15.276), 3) 


    
# current = [(410 - x * 0.8) for x in range(0, 214, 2)]
# watt = list(map(current_to_watt,current))
# watt.reverse()
# indexes = [75,73,71,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
# for i in indexes:
#     del watt[i]

# fig, ax = plt.subplots()

# ax.plot(watt, std_deviations, '-o',lw=1, c='firebrick', alpha=0.75)
# ax.scatter(watt, std_deviations, s=30, facecolors='white', edgecolors='firebrick', zorder=3, alpha=0.75)
# ax.tick_params(axis='both', which='major', labelsize=size)
# ax.set_xlabel('Power (mW)', labelpad=15)
# ax.set_ylabel('$\sigma$', labelpad=15)
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='in')
# plt.tight_layout()
# plt.show()