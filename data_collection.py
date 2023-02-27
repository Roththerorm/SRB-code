# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:44:49 2022

@author: Nicolas Pessoa
"""

# Import necessary libraries

import time
import pyautogui
import pyperclip
import pandas as pd
import pygetwindow as pw 
from pathlib import Path
from datetime import date
from pyscreeze import center

############################ Global variables ############################

low_limit = 980         # Minimum wavelength range of data
high_limit = 1060        # Maximum wavelength range of data
samples = 5000          # number of total samples in one run
n_collection = 1        # Number of collections (different V)
rounds = 1              # Number of rounds the experiment is ran
start = 520              # Begin voltage of collection
step = 1               # Step size of collection
run_time = 700          # Time in seconds of one collection

in_path = r'D:\LaserYb\Medidas Espectrometro'
today = date.today().strftime('%d_%m_%Y')  # Date of collection
hour = time.strftime("%H_%M", time.localtime())

##########################################################################


def spec_collection(copy_button, temp_path, c, r):
    
    for i in range(0, samples):
    
        # Click in the button location
        pyautogui.click(copy_button)
        
        # Save and modify the data from clipboard
        data = pyperclip.paste()
        data = data.replace(",",".")
        
        # This line transforms a single multiple line string into a list of strings
        data_list = [y for y in (x.strip() for x in data.splitlines()) if y]
    
        # Create a data frame with the content
        df = pd.DataFrame(data_list, columns=['W_I'])
        df[['Wavelengths', 'Intensity']] = df['W_I'].str.split('\t', expand=True)
        df.drop('W_I', inplace=True, axis=1)
        df = df.astype(float)
        
        interval = (df['Wavelengths'] >= low_limit) & \
             (df['Wavelengths'] < high_limit + 1)
        
        df[interval].to_csv(f'{temp_path}\\data_{i}.csv', index=False,
                             header=True, sep=' ')
        
        p = round((((i + 1) * 100) / samples), 4)
        print(f'Coleta {c}.{r}: {p:.2f} % concluído')


# Maximize Ocean Optics window (only works if the window was the last minimized window)
pw.getWindowsWithTitle("Ocean Optics SpectraSuite")[0].maximize()

# counter = 0

# while counter == 0:
    
#     pw.getWindowsWithTitle("Optical Power Monitor")[0].maximize()
#     record_location = None

#     while record_location is None:
        
#         # It takes sometime to find the image since it needs to be perfect pixel matching
#         record_location = pyautogui.locateOnScreen(
#         r'C:\Users\nicol\Desktop\SRB code\Images\record_button.png',
#         grayscale = True)

#     # Find the location of the button
#     record_button = center(record_location)
#     pyautogui.click(record_button)
#     time.sleep(.5)
#     pw.getWindowsWithTitle("Optical Power Monitor")[0].minimize()
#     counter =+ 1


# pw.getWindowsWithTitle("Optical Power Monitor")[0].maximize()

# save_location = None

# while save_location is None:
    
#     # It takes sometime to find the image since it needs to be perfect pixel matching
#     save_location = pyautogui.locateOnScreen(
#     r'C:\Users\nicol\Desktop\SRB code\Images\save_button.png',
#     grayscale = True)

# # Find the location of the button
# save_button = center(save_location)
# pyautogui.click(save_button)
# pyautogui.write(f'sample{1}_{1}ma.csv')
# pyautogui.press(['enter', 'enter'])
# print('save done')

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

for i in range(n_collection):

    for j in range(rounds):
    
        Path(rf'{in_path}\{today}\{hour}\{(start + step * i)}_{j}').mkdir(parents=True, exist_ok=True)
        
        # Path where the file was created
        path = rf'{in_path}\{today}\{hour}\{(start + step * i)}_{j}'
        
        # Just to know how long the code takes to collect all data
        start_time = time.time()
        c_time = time.ctime()
        
        spec_collection(copy_button,path, i + 1, j + 1)
        
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
