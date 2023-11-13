'''
Code to convert data from another experiment to test if the current algorithm works.
Data is from the Ultrashort Pulses Lab.
created by: Nícolas Pessoa 
'''

# Import necessary libraries
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Define the directory path where CSV files are located
path = r'D:\150mmes'

# Use Path to create an iterator for CSV files in the directory
files = Path(f'{path}').glob('*.csv')

# Use glob to get a list of all CSV files, then sort them by filename length
all_files = glob.glob(os.path.join(f"{path}", "*.csv"))
all_files.sort(key=len)

# Create a generator that reads each CSV file and yields a DataFrame
df_from_each_file = (pd.read_csv(f) for f in all_files)

# Concatenate the DataFrames along the columns (axis=1) to create one large DataFrame
concatenated_df = pd.concat(df_from_each_file, axis=1, ignore_index=True)

# Create a numpy array containing a range of values for the 'Wavelength' column
wave = np.arange(0, len(concatenated_df))
column_one = pd.DataFrame(wave, columns=['Wavelength'])

# Concatenate the 'Wavelength' column with the existing DataFrame
final_concatenated = pd.concat([column_one, concatenated_df], axis=1)

# Convert the final DataFrame to a numpy array
all_data_array = final_concatenated.to_numpy()

# Save the numpy array as a .npy file
np.save(r'D:\b_data\b_data_150mm_esp.npy', all_data_array)