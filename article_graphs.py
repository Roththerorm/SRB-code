'''
This script generates a figure describing regions of operation of the Yb-Laser.
It reads the diode data and plots a curve of Power x current.
created by: Nícolas Pessoa
'''


# Import libraries 
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

#Define some global parameters of the graphs (font, size, etc.)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : 14})
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['hatch.linewidth'] = 1.5

# Define the path where to pull the data
path = r'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Diode_potency_data.csv'

# Import data from path
data = pd.read_csv(path, sep=';', decimal=',')

################ Plot data ################

fig, ax = plt.subplots(num=1, figsize=[7.6, 4.3])
ax.plot(data['corrente (mA)'], data['potencia (mW)'], lw=2, color='black', label='Diode output')

# Set positions in the x axis where the transition of regime occurs
x_positions = [min(data['corrente (mA)']), 45, 268, 290, 410, 630, max(data['corrente (mA)'])]

# List of positions to add the text
text_pos = [(156, 300,'CW\nREGIME'),
            (351, 285, 'QUASI\nMODE\nLOCKED\nREGIME'),
            (520, 150, 'STANDARD\nMODE-LOCKED\nREGIME'), 
            (755, 150, 'COMPLEX\nDYNAMICS')]

# Define a list of colors for the axvspan rectangles
colors = ['#CED0C8', '#347B98', '#663399', '#DEC121', '#5EBA1C', '#E12514']

# Define a list of hatches for the axvspan rectangles (empty string means no hatch)
hatches = ['', '', '\\\\', '', '', '']

# Loop through the x_positions list to create axvspan rectangles
for i in range(len(x_positions) - 1):
    ax.axvspan(x_positions[i], x_positions[i + 1], color=colors[i], alpha=0.4, hatch=hatches[i])

# Loop through x-coordinates and add vertical lines
for x in x_positions[1:-1]:
    ax.axvline(x=x, color='black')

# Loop through text positions to add text
for x, y, txt in text_pos:
    ax.text(x, y, txt, fontsize=12, color='black', ha='center')

# Add extra text for the first axvspan 
ax.text(18, 65, 'NO SIGNIFICANT SPECTRUM DETECTED', fontsize=10,rotation='vertical')

################ Plot configuration ################

ax.set_xlim(left=1, right=data['corrente (mA)'].max())
ax.set_ylim(bottom=0,top=data['potencia (mW)'].max())

ax.set_xlabel('Current (mA)', labelpad=15)
ax.set_ylabel('Power (mW)', labelpad=15)
ax.tick_params(axis='both', direction='in')

ax.legend(loc='best')
plt.tight_layout()
plt.savefig(rf'C:\Users\nicol\Desktop\Figuras (Marcio)\regions of operations.pdf')

