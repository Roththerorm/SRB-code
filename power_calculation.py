import pandas as pd
import numpy as np


def compute_power(f_data, data, trapezoid):
    # Extract the values from the 'Wavelengths' column of f_data and flatten them into a 1D array
    wave_array = f_data['Wavelengths'].values.flatten()
    
    # Create a list of lists called 'intensities' containing the values of each column of f_data,
    # except the 'Wavelengths' column, in the form of a list
    intensities = [f_data.iloc[:, i + 1].values.tolist() for i in range(len(data[0]) - 1)]
    
    # Convert the 'intensities' list into a numpy array
    intensities = np.array(intensities)
    
    # Apply the 'trapezoid' function along the rows of the 'intensities' array, using the 'wave_array' as an input
    power = np.apply_along_axis(trapezoid, 1, intensities, wave_array)
    
    # Divide the resulting array by 10 and round to two decimal points
    power = np.round(power / 10, 3)
    
    # Return the resulting array
    return power