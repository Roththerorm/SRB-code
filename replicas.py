import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines
import seaborn as sns
import matplotlib
import time
import math
import zoom_effect
from matplotlib.patches import ConnectionPatch
from spectrum_animation import plot_animation
from power_calculation import compute_power
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import trapezoid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import graphs
import os
# Imported full libraries
import os
import math
import random
import cycler
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Imported modules from libraries and Python codes
from cProfile import label
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from functools import partial

# import graphs
import graphs
from power_calculation import compute_power
from spectrum_animation import plot_animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.transforms as mtransforms
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.ticker as ticker


pico = 0                # Pick taken in the samples of power
percentage = 0.5        # Filter used to select the points
num_points = 3000     # Number of points to calculate Parisi Coefficient
total_points = 3000     # Total number of replicas
first_binary_data = 408
last_binary_data = 408
step = 2
mod_q = []

k = 240
l = 1

size = 12.5
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : size})
matplotlib.rcParams['axes.linewidth'] = 1.


def plottable_3d_info(df: pd.DataFrame, min_value, max_value):
    """
    Transform Pandas data into a format that's compatible with
    Matplotlib's surface and wireframe plotting.
    """
    index = df.index
    columns = df.columns

    x, y = np.meshgrid(np.arange(len(columns)), f_data.iloc[min_value:max_value,0])
    z = np.array([[df[c][i] for c in columns] for i in index])
    
    xticks = dict(ticks=np.arange(len(columns)), labels=columns)
    yticks = dict(ticks=np.arange(len(index)), labels=index)
    
    return x, y, z, xticks, yticks
    
path = r'D:\LaserYb\Medidas Espectrometro\mes 02_23\17_02_2023\binary_data'

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
    f_data['Wavelengths'] = f_data['Wavelengths'] + 2.5

    min_value, max_value = 120, 220
    x, y, z, xticks, yticks = plottable_3d_info(f_data.iloc[min_value:max_value,1:3000], min_value, max_value)
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    axes.plot_surface(x, y, z, cmap='icefire', rcount=1000, linewidth=0, antialiased=False)
    axes.invert_xaxis()
    axes.view_init(24, 30)
    axes.set_xlabel('N', labelpad=10)
    axes.set_ylabel('Wavelength (nm)', labelpad=10)
    axes.zaxis.set_rotate_label(False)
    axes.set_zlabel('Intensity (arb. units)', labelpad=10, rotation = 90) 
    plt.tight_layout()
    plt.show()


