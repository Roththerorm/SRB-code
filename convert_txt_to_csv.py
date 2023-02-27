import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

path = r'D:\150mmes'

files = Path(f'{path}').glob('*.csv')
all_files = glob.glob(os.path.join(f"{path}", "*.csv"))
all_files.sort(key=len)
df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df = pd.concat(df_from_each_file, axis=1, ignore_index=True)

wave = np.arange(0, len(concatenated_df))
column_one = pd.DataFrame(wave, columns=['Wavelength'])
final_concatenated = pd.concat([column_one, concatenated_df], axis=1)
all_data_array = final_concatenated.to_numpy()
    
np.save(r'D:\b_data\b_data_150mm_esp.npy', all_data_array)
