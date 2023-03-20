# Import necessary libraries
import os
import re
import glob
import time
import numpy as np
import pandas as pd

# Import modules
from pathlib import Path

# Define Functions 
def eliminate_non_numeric(path):
    
    """
    Eliminate elements from a list that do not start with a digit.

    Parameters:
        path (str): The path to a directory.

    Returns:
        list: A list of elements that start with a digit.
    """
    
    # Check if the path exists and is a directory
    if not os.path.isdir(path):
        raise ValueError(f'{path} is not a valid directory')

    # Get the list of elements in the directory
    files = os.listdir(path)

    # Use a list comprehension to create a new list containing only the elements that start with a digit
    numeric_lst = [element for element in files if re.match(r'^\d', element)]

    # Return the new list
    return numeric_lst

####################### CODE #################################

# Use the path that only contains the data of collection
path = r'D:\LaserYb\Medidas Espectrometro\mes 12\22_12_2022'

# Select the elements that only star with a number 
# (DO NOT CREATE ANY FOLDER THAT STARTS WITH A NUMBER INSIDE THE FILE)
f_names = eliminate_non_numeric(path)

# Create a file to save the converted data
Path(rf'{path}\binary_data').mkdir(parents=True, exist_ok=True)

# Loop over all elements
for i in range(len(f_names)):
    
    # Make a subset with the elements inside the file
    sub_list = eliminate_non_numeric(os.path.join(path, f_names[i]))
    for j in range(len(sub_list)):

        files = Path(rf'{path}\{f_names[i]}\{sub_list[j]}').glob('*.csv')
        all_files = glob.glob(os.path.join(rf"{path}\{f_names[i]}\{sub_list[j]}", "*.csv")) 
        all_files.sort(key=len)
    
        n_files = len(all_files) - 1

        st = time.time()

        df_from_each_file = (pd.read_csv(f, delimiter=' ') for f in all_files)
        concatenated_df   = pd.concat(df_from_each_file, axis=1, ignore_index=True)
        concatenated_df.drop((concatenated_df.columns[k] for k in range(1,2*n_files+1) if k % 2 == 0),
        axis=1, inplace=True)

        all_data_array = concatenated_df.to_numpy()
    
        np.save(rf'{path}\binary_data\b_data_{sub_list[j]}.npy', all_data_array)

        print(f'Data convertida {i+1} de {len(f_names)}. \n Sub {j} de {len(sub_list)} \n Tempo total {time.time() - st} segundos')