# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:44:49 2022

@author: Nicolas Pessoa
"""

# Import necessary libraries

import time
import pyvisa
import pyautogui
import pyperclip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygetwindow as pw 
from pathlib import Path
from datetime import date
from pyscreeze import center

############################ Global variables ############################

low_limit = 990         # Minimum wavelength range of data
high_limit = 1060        # Maximum wavelength range of data
samples = 3000         # number of total samples in one run
n_collection = 1        # Number of collections (different V)
rounds = 1              # Number of rounds the experiment is ran
start = 408              # Begin voltage of collection
step = 1               # Step size of collection
run_time = 1          # Time in seconds of one collection


osc_addr = "USB0::0x0957::0x175D::MY50340840::INSTR"

channel = 1
total_points = 5000

in_path = r'D:\LaserYb\Medidas Espectrometro'
in_path_osc = r'D:\LaserYb\Medidas Osciloscópio'
today = date.today().strftime('%d_%m_%Y')  # Date of collection
hour = time.strftime("%H_%M", time.localtime())

##########################################################################


def channel_data(instrument, channel, n_points_mode, n_points):
    
    instrument.write(f':DIGitize [CHANnel{channel}:]')
    instrument.write(f':WAVeform:SOURce CHANnel{channel}')
    instrument.write(f':WAVeform:FORMat ASCii')
    instrument.write(f':WAVeform:POINts:MODE {n_points_mode}')
    instrument.write(f':WAVeform:POINts {n_points}')
    instrument.write(':WAVeform:DATA?')

    values = [n for n in instrument.read().replace(' ', ',').split(',')]
    data_list = list(map(float,filter(None,values[1:])))

    return data_list

def channel_config(instrument, channel, n_points_mode, n_points):
    
    instrument.write(f':DIGitize [CHANnel{channel}:]')
    instrument.write(f':WAVeform:SOURce CHANnel{channel}')
    instrument.write(f':WAVeform:FORMat ASCii')
    instrument.write(f':WAVeform:POINts:MODE {n_points_mode}')
    instrument.write(f':WAVeform:POINts {n_points}')

def time_vector(instrument, channel, n_points_mode, n_points):

    channel_config(instrument, channel, n_points_mode, n_points)
    
    instrument.write(':SINGle')
    instrument.write(':WAVeform:XORigin?')
    x_0 = float(instrument.read())

    instrument.write(':WAVeform:XINCrement?')
    dx= float(instrument.read())

    instrument.write(':RUN')
    array = np.linspace(x_0, x_0 + n_points * dx, n_points)
    
    return array

def spec_osc_collection(instrument, copy_button, temp_path_osc, temp_path_spec, osc_x_vector, c, r):
    
    for i in range(0, samples):

        # Save data from Oscilloscope
        instrument.write(':STOP')
        instrument.write(':WAVeform:DATA?')
        values = [n for n in instrument.read().replace(' ', ',').split(',')]
        # Click in the button location
        pyautogui.click(copy_button)
        
        instrument.write(':RUN')
        # Save and modify the data from clipboard 
        data = pyperclip.paste()
        data = data.replace(",",".")
        
        # This line transforms a single multiple line string into a list of strings
        data_list = [y for y in (x.strip() for x in data.splitlines()) if y]

        osc_data = list(map(float,filter(None,values[1:])))
    
        # Create a data frame with the content of the Spectrometer
        df = pd.DataFrame(data_list, columns=['W_I'])
        df[['Wavelengths', 'Intensity']] = df['W_I'].str.split('\t', expand=True)
        df.drop('W_I', inplace=True, axis=1)
        df = df.astype(float)

        # Create a data frame with the content of the Oscilloscope
        if len(osc_x_vector) != len(osc_data):
            osc_data = osc_data[:-1]
        else:
            pass

        df_2 = pd.DataFrame({'Time' : osc_x_vector, 'Voltage' : osc_data})
        
        interval = (df['Wavelengths'] >= low_limit) & (df['Wavelengths'] < high_limit + 1)
        
        # Save data 
        df[interval].to_csv(f'{temp_path_spec}\\data_{i}.csv', index=False, header=True, sep=' ')
        df_2.to_csv(f'{temp_path_osc}\\osc_data_{i}.csv', index=False, header=True, sep=' ')
        
        
        p = round((((i + 1) * 100) / samples), 4)
        print(f'Coleta {c}.{r}: {p:.2f} % concluído')
        # time.sleep(0.1)

# Maximize Ocean Optics window (only works if the window was the last minimized window)
pw.getWindowsWithTitle("Ocean Optics SpectraSuite")[0].maximize()

# wait_time = int(time.strftime('%M'))
# loop = int(time.strftime('%M'))

# while loop != wait_time + 1:
#     loop = int(time.strftime('%M'))

# Find the data copy button on SpectraSuite.
copy_location = None

while copy_location is None:
    
    # It takes sometime to find the image since it needs 
    # to be perfect pixel matching
    copy_location = pyautogui.locateOnScreen(
    r'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Programa Espectrômetro\button1.png',
    grayscale = True)
    if copy_location == None:
        copy_location = pyautogui.locateOnScreen(
        r'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Programa Espectrômetro\button2.png',
        grayscale = True)

# Find the location of the button
copy_button = center(copy_location)

time.sleep(10)
t_time = time.time()

Path(rf'{in_path}\{today}\{hour}\Information').mkdir(parents=True, exist_ok=True)

rm = pyvisa.ResourceManager()
inst_addr_list = rm.list_resources()

if inst_addr_list.count(osc_addr) >= 1:
    pass
else:
    print(f'Error: Oscilloscope address "{osc_addr}" does not exist')

osc_inst = rm.open_resource(osc_addr)
x_vector = time_vector(osc_inst, channel, 'MAXimum', total_points)
x_vector = x_vector[:-1]

for i in range(n_collection):

    for j in range(rounds):
    
        Path(rf'{in_path}\{today}\{hour}\{(start + step * i)}_{j}').mkdir(parents=True, exist_ok=True)
        Path(rf'{in_path_osc}\{today}\{hour}\{(start + step * i)}_{j}').mkdir(parents=True, exist_ok=True)
        
        # Path where the file was created
        path_spec = rf'{in_path}\{today}\{hour}\{(start + step * i)}_{j}'
        path_osc = rf'{in_path_osc}\{today}\{hour}\{(start + step * i)}_{j}'
        
        # Just to know how long the code takes to collect all data
        start_time = time.time()
        c_time = time.ctime()
        
        channel_config(osc_inst, channel, 'MAXimum', total_points)

        spec_osc_collection(osc_inst, copy_button, path_osc, path_spec, x_vector, i + 1, j + 1)

        end_time = (time.time() - start_time)
        
        print('\n')
        print("Time taken: --- %s seconds ---" % (end_time))
        
        if end_time < run_time:
            time.sleep(run_time - end_time)
        else:
            pass
        
        with open (rf'{in_path}\{today}\{hour}\Information\info_{i+1}_{j}.txt', 'w') as f:
            f.write('Important information:\n')
            f.write('Collection number %s' % (i+1))
            f.write('Begin: %s \n' % (c_time))
            f.write('End: %s \n' % (time.ctime()))
            f.write('Range: %s to %s \n' % (low_limit, high_limit))
            f.write('Voltage (mV): %s \n' % (start + step * i))
            f.write('Samples: %s \n' % samples)
            f.write('Time taken: --- %s seconds ---' % end_time)
        
        print('\n')

pw.getWindowsWithTitle("Ocean Optics SpectraSuite")[0].minimize()

print("Total time --- %s seconds ---" % (time.time() - t_time))
