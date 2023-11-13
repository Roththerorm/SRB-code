import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
import pyvisa
import time 
from seabreeze.spectrometers import Spectrometer

low_limit = 990                 # Minimum wavelength range of data
high_limit = 1070               # Maximum wavelength range of data
large_samples = 150000          # number of total samples in one run
intermediate_samples = 3000
interval = 5
n_collection = 1              # Number of collections (different V)
rounds = 1                      # Number of rounds the experiment is ran
start = 140                     # Begin voltage of collection
step = -3                       # Step size of collection
run_time = 1                    # Time in seconds of one collection
initial_volt_value = 140        # in mV

in_path = r'D:\LaserYb\Medidas Espectrometro'

today = date.today().strftime('%d_%m_%Y')  # Date of collection
hour = time.strftime("%H_%M", time.localtime())


def spec_collection(spectrometer, temp_path_spec, samples, c, r):

    for i in range(0, samples):
        
        wavelengths = spectrometer.wavelengths()
        intensity = spectrometer.intensities()
        df = pd.DataFrame({'Wavelengths' : wavelengths, 'Intensity' : intensity})
        interval = (df['Wavelengths'] >= low_limit) & (df['Wavelengths'] < high_limit + 1)
        df[interval].to_csv(f'{temp_path_spec}\\data_{i}.csv', index=False, header=True, sep=' ')
        
        if i % 10 == 0:
            p = round((((i + 1) * 100) / samples), 4)
            print(f'Coleta {c}.{r}: {p:.2f} % concluído')


t_time = time.time()
Path(rf'{in_path}\{today}\{hour}\Information').mkdir(parents=True, exist_ok=True)

rm = pyvisa.ResourceManager()
wave_generator = rm.open_resource('USB0::0x0957::0x2C07::MY52812459::INSTR')
wave_generator.write('FUNCtion DC') # type: ignore
wave_generator.write(f'VOLTage:OFFS +{initial_volt_value / 1000}') # type: ignore
wave_generator.write('OUTPut1 1') # type: ignore


spec = Spectrometer.from_first_available()
spec.integration_time_micros(9000)

print('Start program')
time.sleep(5)

for i in range(n_collection):

    wave_generator.write(f'VOLTage:OFFS +{(initial_volt_value + step * i) / 1000}') # type: ignore
    wave_generator.write('OUTPut1 1') # type: ignore
    
    if i % interval == 0:
        total_replicas = large_samples
    else:
        total_replicas = intermediate_samples
    
    for j in range(rounds):
    
        Path(rf'{in_path}\{today}\{hour}\{(start + step * i)}_{j}').mkdir(parents=True, exist_ok=True)
        path_spec = rf'{in_path}\{today}\{hour}\{(start + step * i)}_{j}'

        # Just to know how long the code takes to collect all data
        start_time = time.time()
        c_time = time.ctime()

        spec_collection(spec, path_spec, total_replicas, i + 1, j + 1)

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
            f.write('Samples: %s \n' % total_replicas)
            f.write('Time taken: --- %s seconds ---' % end_time)
        
        print('\n')

print("Total time --- %s seconds ---" % (time.time() - t_time))

