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
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['hatch.linewidth'] = 1.5

fig, ax = plt.subplots(num=1, figsize=[7.6, 4.3])

ax.plot(data['corrente (mA)'], data['potencia (mW)'], lw=2, color='black',
label='Diode output')


x_positions = [min(data['corrente (mA)']),
            45, 268, 290, 410, 630, max(data['corrente (mA)'])]



# List of positions to add the text
text_pos = [(156, 300,'CW\nREGIME'),
(351, 285, 'MULTI\nMODE\nREGIME'),
(520, 150, 'MODE-LOCKED\nREGIME'), 
(755, 150, 'COMPLEX\nDYNAMICS')]

ax.axvspan(x_positions[0], x_positions[1], color='#CED0C8', alpha=0.4)
ax.axvspan(x_positions[1], x_positions[2], color='#347B98', alpha=0.4)
ax.axvspan(x_positions[2], x_positions[3], color='#663399', alpha=0.7, hatch="\\\\\\")
ax.axvspan(x_positions[3], x_positions[4], color='#DEC121', alpha=0.4)
ax.axvspan(x_positions[4], x_positions[5], color='#5EBA1C', alpha=0.4)
ax.axvspan(x_positions[5], x_positions[6], color='#E12514', alpha=0.4)


ax.text(18, 65, 'NO SIGNIFICANT SPECTRUM DETECTED', fontsize=10,rotation='vertical')



# Loop through x-coordinates and add vertical lines

x_positions_new = [45, 268, 290, 410, 630]
for x in x_positions_new:
    ax.axvline(x=x, color='black')

for x, y, txt in text_pos:
    ax.text(x, y, txt, fontsize=10, color='black',
    ha='center')


ax.set_xlim(left=1, 
            right=max(data['corrente (mA)']))
ax.set_ylim(bottom=0,
            top=max(data['potencia (mW)']))
ax.set_xlabel('Current (mA)', labelpad=15)
ax.set_ylabel('Power (mW)', labelpad=15)
ax.tick_params(axis='both', direction='in')
ax.legend(loc='best')
plt.tight_layout()
plt.savefig(rf'C:\Users\nicol\OneDrive\Documentos\1 - Faculdade\Metrologia\Escrita\Universal manuscript template for Optica Publishing Group journals\figures\regions_operation.pdf')

