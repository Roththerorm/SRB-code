# from seabreeze.spectrometers import Spectrometer
# from spectrum_animation import plot_animation
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# spec = Spectrometer.from_first_available()
# spec.integration_time_micros(9000)

# wavelengths = spec.wavelengths()

# init = time.time()
# array = [0] * 3000

# for i in range(3000):
#     spec.integration_time_micros(4500)
#     array[i] = spec.intensities()
    

# print((time.time() - init) / 60)

# array = np.transpose(array)
# columns = [f'Replica_{i}' for i in range(1, 3000 + 1)]
# data = pd.DataFrame(array, columns=columns)
# data.insert(0, "Indices", wavelengths, True)

# plot_animation(data, 0, 4000, 0, 15000, 0, 'n', 20, False)

import pyvisa
import time

rm = pyvisa.ResourceManager()
lista = rm.list_resources()

print(lista)

# wave_generator.write('FUNCtion DC')
# wave_generator.write(f'VOLTage:OFFS +{initial_value}')
# wave_generator.write('OUTPut1 1')

# step = 0.1
# time.sleep(3)

# for i in range(1, 10):
#     wave_generator.write(f'VOLTage:OFFS +{initial_value - step * i}')
#     wave_generator.write('OUTPut1 1')

#     time.sleep(1)



