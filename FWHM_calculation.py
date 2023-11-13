# Imported full libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spectrum_animation import plot_animation
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.signal import find_peaks
from scipy.constants import c
from scipy import signal
from scipy import optimize
from numba import jit
import matplotlib
import math


############# GLOBAL VARIABLES #############
              
first_binary_data = 408
last_binary_data = 409
step = 1

############# MAIN CODE #############

size = 12
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : size})
matplotlib.rcParams['axes.linewidth'] = 1.

l = 0

def fitting_curve(x, a, b, x_0):
    return a * np.exp(-(x-x_0) ** 2 * b ** 2) + 0.8
    # return a * ((1 / np.cosh((x-x_0) * b))) ** 2 + 1
    



for k in range(first_binary_data, last_binary_data, step):
    
    print(f'Current file ({k}) ({((k-first_binary_data)//step) + 1} of {int((last_binary_data - first_binary_data)/2)}). Round: {l}')
    
    path_osc = rf'D:\LaserYb\Medidas Osciloscópio\29_09_2023\binary_data'

    filename_osc = f'b_data_{k}_{l}.npy'
    filepath_osc = os.path.join(path_osc, filename_osc)

    # Use os.path.exists to check if the file exists
    if not os.path.exists(filepath_osc):
        print(f'Error: File "{filepath_osc}" does not exist')
    else:
        # Use the with statement to open the file
        with open(filepath_osc, 'rb') as f:
            loading = np.load(f)
            data = np.delete(loading, 1, 1) # Check the size, there's a bug in the first column of the data, maybe is due to the transformation. do not know
  
        # Use list comprehension to generate the list of column names
        columns = ['Time'] + [f'Voltage_{i}' for i in range(1, len(data[0]))]
        
        # Use f-strings to format the column names
        f_data = pd.DataFrame(data, columns=columns)
        
        # Remove delay of the oscilloscope and set the time in [s]
        f_data['Time'] = (f_data['Time'] - 2 * f_data.iat[0,0]) * 1e6
        
        # Normalize all data to be in the scale 1:8 
        f_data.iloc[:,1:] -= f_data.iloc[:,1:].min()
        f_data.iloc[:,1:] /= f_data.iloc[:,1:].max() * (1/8)
        
        # Obtain the average period spacing among all autocorrelation data

        overall_average_distances = []

        for i in range(1, len(f_data.columns)):
            peaks, _ = find_peaks(f_data.iloc[:, i], height=1.9)
            
            # Calculate the average distance using NumPy
            distances = np.diff(peaks)
            average_distance = np.mean(distances)
            
            overall_average_distances.append(average_distance)

        # Calculate the overall average of all columns
        overall_average = np.mean(overall_average_distances)

        # Calibrate time axis. the period of the autocorrelation fringes represent, in scale, the optical period T = λ_0 / c. For our case,  λ_0 = 1025 nm.
        time_length = f_data.shape[0]

        # Set the step between points
        dt = 1025e-09 / (c * overall_average)

        # Obtain the extremes of the interval. Remember that we set the the center of the envelope as t = 0.
        reference_time = dt * time_length / 2
        
        # Crate the calibrated time and replace for the old one. There's a scale factor of 10^13 so that the axis are single digits and represents 10^2 fs. Scale factors above or below this prevents fitting calculations.
        new_time = np.linspace(-reference_time, reference_time, time_length)
        f_data['Time'] = new_time * 1e13

        
        # Select interval of data to make the fit. Is necessary to get only the main part of the data. Extremes will mess up the fit
        fwhm_values = []
        sigma_values = []

        threshold = 2.0

        # Use a dictionary comprehension to find indices for each column
        # above_threshold = {col: f_data.query(f'{col} > {threshold}').index for col in f_data.columns}

        for i in range(1, 2):
            
            limits = [1100, 3800]
            #limits = above_threshold[f'{f_data.columns[i]}'].tolist()
            x_values = f_data.iloc[limits[0]:limits[-1], 0].to_numpy()
            y_values = f_data.iloc[limits[0]:limits[-1], i].to_numpy()

            # Use NumPy array operations for efficient filtering. We're selecting data above 1 to do the correct fit
            mask = (y_values >= 1) & (y_values > np.roll(y_values, 1)) & (y_values > np.roll(y_values, -1))
            filtered_x = x_values[1:-1][mask[1:-1]]
            filtered_y = y_values[1:-1][mask[1:-1]]

            x_data = filtered_x
            y_data = filtered_y
            
            params, params_covariance = optimize.curve_fit(fitting_curve, x_data, y_data, p0=[8, 1, 0])
            
            x = f_data.iloc[:, 0].to_numpy()
            y = fitting_curve(x, *params)  # Use *params to unpack the parameter values

            max_y = np.max(y)
            max_x = x[np.argmax(y)]
            half_max = max_y / 2

            # Use a single line to calculate FWHM
            indices_above_half_max = np.where(y >= half_max)[0]
            fwhm = x[indices_above_half_max[-1]] - x[indices_above_half_max[0]]
            
            # Divide by the Gaussian deconvolution factor (sqrt(2)) and proper scale facotr
            fwhm = (fwhm / np.sqrt(2)) * 1e2

            sigma = (params[1] / 2) * 1e2
            
            fwhm_values.append(fwhm)
            sigma_values.append(sigma)
            
            fig, ax = plt.subplots()
        
            ax.plot(f_data.iloc[:,0] * 1e2, f_data.iloc[:,i], lw=0.2, c='maroon', label='Autocorrelation')
            ax.plot(x * 1e2, fitting_curve(x * 1e2, params[0], params[1] / 1e2, params[2]  * 1e2), c='black', ls='-', lw=1.5, label='Gaussian fit')
            
            ax.arrow(x[indices_above_half_max[0]] * 1e2, half_max, (x[indices_above_half_max[-1]] - params[2]) * 2e2, 0, head_width= 0.2, head_length= 1.5 * 0.2 * 1e2, length_includes_head=True, color='black', zorder=3)

            ax.arrow(x[indices_above_half_max[-1]] * 1e2, half_max, (x[indices_above_half_max[0]] + params[2]) * 2.8e2, 0, head_width= 0.2, head_length= 1.5 * 0.2 * 1e2, length_includes_head=True, color='black', zorder=3)

            ax.text(x[indices_above_half_max[-1]] * 1e2 + 20, half_max, f'FWHM ≈ {math.floor(fwhm)} fs', fontsize = 10)

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlabel('time (fs)', labelpad=15)
            ax.set_ylabel('Amplitude (arb. units)', labelpad=15)
            ax.tick_params(axis='both', which='both', direction="in", labelsize=size)
            plt.tight_layout()
            plt.legend()
            plt.show()
            #plt.savefig(rf'D:\LaserYb\Resultado de Tratamento de Dados\Autocorrelação\29_09_23\fig_{i}.png')
            #plt.close()
            
            print(f'Done {i}')


        # df = pd.DataFrame(fwhm_values)
        # df_2 = pd.DataFrame(sigma_values)


        # # Save the DataFrame to a CSV file
        # df.to_csv('fwhm_array_3.csv', index=False, header=False)
        # df.to_csv('sigma_array_3.csv', index=False, header=False)







        # # maxima = f_data.max('index')


        # # peaks, _ = find_peaks(f_data.iloc[:,1], height=(maxima[1]/2))

        # # print(f'largura é {(len(peaks)) * 1e15 * (1025e-09/c)} fs')


        # # fig, ax = plt.subplots()
        
        # # ax.plot(f_data.iloc[:,0] * 1e2, f_data.iloc[:,1], lw=0.2, c='maroon', label='Autocorrelation')
        # # ax.plot(x * 1e2, fitting_curve(x * 1e2, params[0], params[1] / 1e2), c='black', ls='-', lw=1.5, label='Gaussian fit')

        
        # # ax.arrow(x[indices_above_half_max[0]] * 1e2, half_max, x[indices_above_half_max[-1]] * 2e2, 0, head_width= 0.2, head_length= 1.5 * 0.2 * 1e2, length_includes_head=True, color='black', zorder=3)

        # # ax.arrow(x[indices_above_half_max[-1]] * 1e2, half_max, x[indices_above_half_max[0]] * 2e2, 0, head_width= 0.2, head_length= 1.5 * 0.2 * 1e2, length_includes_head=True, color='black', zorder=3)

        # # ax.text(x[indices_above_half_max[-1]] * 1e2 + 20, half_max, 'FWHM ≈ 260 fs', fontsize = 10)


        # # # ax.plot(f_data.iloc[peaks,0] * 1e2, f_data.iloc[peaks,1], 'x', color='black')
        
        # # ax.set_xlabel('time (fs)', labelpad=15)
        # # ax.set_ylabel('Amplitude (arb. units)', labelpad=15)
        # # ax.tick_params(axis='both', which='both', direction="in", labelsize=size)

        # # plt.tight_layout()
        # # plt.legend()
        # # plt.show()


        # #plt.plot(x_values, fitting_curve(np.array(x_values), params[0], params[1]), ls='--')
        
        # # fig, ax = plt.subplots()
        # #ax.plot(f_data_spec.iloc[:,0], f_data_spec.iloc[:,1], lw=0.2)        
    


        # # plot_animation(f_data, 0, 5000, -.25, 1.75, 0, 'n', 20, False)
