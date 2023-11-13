import numpy as np
import pandas as pd
import random
from scipy.signal import find_peaks

def remove_outliers(power_samples, percentage, pico, num_points, total_points, df):

    # np.histogram to find the histogram peaks and apply Square-root choice for the number of bins
    root_bins = int(np.sqrt(len(power_samples + 1)))
    histogram, histogram_bins = np.histogram(power_samples, bins= 2 * root_bins)
    peaks, _ = find_peaks(histogram, height=10, prominence=30, distance=10)

    # List comprehension to create the indices_inside and indices_outside lists
    lim_max = histogram_bins[peaks][pico] + percentage * histogram_bins[peaks][pico]
    lim_min = histogram_bins[peaks][pico] - percentage * histogram_bins[peaks][pico]
    indices_outside = [i for i in range(len(power_samples)) if power_samples[i] > lim_max or power_samples[i] < lim_min]
    indices_inside = [i for i in range(len(power_samples)) if i not in indices_outside ]

    # Stop the program if the percentage chosen selected less than the number of desired points
    if len(indices_outside) > total_points - num_points:
        raise ValueError(f'The percentage applied ({percentage*100}%) results in less than {num_points} replicas')

    # Show how many points were excluded
    discarded = len(indices_inside) - num_points
    
    # Create a new list that only includes elements that are not equal to 0 in the outside_indices
    indices_outside_no_zero = [x for x in indices_outside if x != 0]
    
    # Modify the list of incide indices to include the element zero. If already in, it deletes duplicates
    indices_inside_zero = [0] + list(set(indices_inside + [0]))[1:]

    # Loop condition to ensure that the length of the points_in list is equal to the desired number of points
    while len(indices_inside_zero) > num_points:
        
        # Select a random point from the indices_inside list
        random_index = random.sample(indices_inside_zero, 1)[0]
        
        # Remove the selected point from both the indices_inside_zero and add to the indices_outside_no_zero list
        if random_index == 0:
            
            # this makes sure we do not exclude the zero element from the list
            continue
        else:
            indices_inside_zero.remove(random_index)
            indices_outside_no_zero.append(random_index)

    # Reverse the order of the elements in the list
    indices_outside_no_zero = indices_outside_no_zero[::-1]

    # Number of elements in the data frame to use as a division in the total sum
    n_div = total_points - len(indices_outside_no_zero)

    # Use pd.DataFrame.drop to remove columns from the data frame that includes the elements
    df.drop((df.columns[i] for i in indices_outside_no_zero), axis=1, inplace=True)
    
    return df, n_div, discarded, lim_max, lim_min