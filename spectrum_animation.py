'''
Created by @ Nicolas Pessoa
20/02/2023

This script generates an animation of the data collected in the spectrometer. It's possible to plot the peaks (needs update to show more than one peak)
'''


import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.animation import FuncAnimation


def spec_animation(i, ax, df, min_wave, max_wave, y_low, y_max, noise, peaks_show):
                
    '''
    i - number of frames in animation. In this case is the total number or replicas
    ax - axis of figure
    f_data - data frame
    min_wave - left wavelength limit
    max_wave - right wavelength limit
    y_low - maximum y value
    y_max - maximum y value
    noise - noise level in the spectrum
    peaks_show - if yes, show the peaks in the animation
    '''
    
    ax.clear()

    size = 12
   
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams.update({'font.size' : size})
    matplotlib.rcParams['axes.linewidth'] = 1.
    
    
    # skip the first column of the data frame since corresponds to the wavelengths
    if i == 0:
        pass
    else:
        if peaks_show == 'y':
        
            spectrum = df.iloc[:, i].values
            peaks, _ = find_peaks(spectrum, height = noise, distance=2, prominence=10)
            
            peak_values = spectrum[peaks]

            if peak_values is not None and len(peaks) > 1:
                
                prominences = _['prominences']

                peak_heights = _['peak_heights']
                
                # sort the peaks by their prominence values
                sorted_peaks = sorted(zip(peaks, prominences, peak_heights), key=lambda x: x[1], reverse=True)

                # select the two peaks with the highest prominences
                top_two_peaks = [sorted_peaks[0][0], sorted_peaks[1][0]]

                top_two_peak_heights = [sorted_peaks[0][2], sorted_peaks[1][2]]


                ax.plot(df.iloc[min_wave:max_wave, 0], df.iloc[min_wave:max_wave, i], lw=1.5, color='firebrick')
                ax.tick_params(axis='both', which='major', labelsize=size)
                ax.plot(df.iloc[top_two_peaks,0], df.iloc[top_two_peaks,i], 'v', color='black', label='peak')

                ax.set_xlabel('Wavelength (nm)', labelpad=15)
                ax.set_ylabel('Intensity (arb. units)', labelpad=15)
                ax.set_ylim(y_low, y_max)
                ax.legend(loc='upper right')
                plt.tight_layout()
           
            else:
                ax.plot(df.iloc[min_wave:max_wave, 0], df.iloc[min_wave:max_wave, i], lw=1.5, color='firebrick')
                ax.tick_params(axis='both', which='major', labelsize=size)
                ax.plot(df.iloc[peaks,0], df.iloc[peaks,i], 'v', color='black', label='peak')

                ax.set_xlabel('Wavelength (nm)', labelpad=15)
                ax.set_ylabel('Intensity (arb. units)', labelpad=15)
                ax.set_ylim(y_low, y_max)
                ax.legend(loc='upper right')
                plt.tight_layout()

        
        elif peaks_show == 'n':
            
            ax.plot(df.iloc[min_wave:max_wave, 0], df.iloc[min_wave:max_wave, i], lw=1.5, color='firebrick')
            ax.tick_params(axis='both', which='major', labelsize=size)
            ax.set_xlabel('Wavelength (nm)', labelpad=15)
            ax.set_ylabel('Intensity (arb. units)', labelpad=15)
            ax.set_ylim(y_low, y_max)
            
            plt.tight_layout()
        
        else:
            print('Insert a valid letter: y or n')


def plot_animation(*args):

    '''
    Arguments order in the function
    args {
        [0] - f_data: data frame
        [1] - min_wave: left wavelength limit
        [2] - max_wave: right wavelength limit
        [3] - y_low: maximum y value
        [4] - y_max: maximum y value
        [5] - noise: noise level in the spectrum
        [6] - peaks_show: if 'y', show the peaks in the animation
        [7] - interval: interval between frames
        [8] - repeat: bool
    }
    '''
    
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams.update({'font.size' : 12})
    matplotlib.rcParams['axes.linewidth'] = 1.
    
    fig, ax = plt.subplots() 
    anim = FuncAnimation(fig, spec_animation, frames=len(args[0].columns), fargs=(ax, args[0], args[1], args[2], args[3], args[4], args[5], args[6]), interval=args[7], repeat=args[8])

    # Function that changes the plt.show break to true and alow to the rest of the code to carry on
    def on_close(event):
        plt.close('all')

    fig.canvas.mpl_connect('close_event', on_close)
    
    plt.show()
