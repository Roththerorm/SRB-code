# Import libraries 
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


# Path
path = r'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Diode_potency_data.csv'

# Import data
data = pd.read_csv(path, sep=';', decimal=',')

# Plot data

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams.update({'font.size' : 12})
matplotlib.rcParams['axes.linewidth'] = 1.


fig, ax = plt.subplots(num=1, figsize=[7.48 , 4.3])

ax.plot(data['corrente (mA)'], data['potencia (mW)'], lw=2, color='black',
label='Diode output')

x_positions = [min(data['corrente (mA)']),
            45, 405, 630, max(data['corrente (mA)'])]

ax.text(15, 150, 'NO SPECTRE DETECTED', fontsize=10
        ,rotation='vertical')

# List of positions to add the text
text_pos = [(202, 300,'\"CW\"\nOPERATION'), 
(517, 150, 'MODE-LOCKED\nOPERATION'), 
(755, 150, 'COMPLEX\nDYNAMICS')]

# ax.axvspan(x_positions[0], x_positions[1], color='#CED0C8', alpha=0.4)
# ax.axvspan(x_positions[1], x_positions[2], color='#347B98', alpha=0.4)
# ax.axvspan(x_positions[2], x_positions[3], color='#5EBA1C', alpha=0.4)
# ax.axvspan(x_positions[3], x_positions[4], color='#E12514', alpha=0.4)


# Loop through x-coordinates and add vertical lines
for x in x_positions:
    ax.axvline(x=x, color='black', linestyle='dashed')

for x, y, txt in text_pos:
    ax.text(x, y, txt, fontsize=10, color='black',
    ha='center')


ax.set_xlabel('Current (mA)', labelpad=15)
ax.set_ylabel('Power (mW)', labelpad=15)
ax.get_tightbbox()
ax.legend(loc='best')

