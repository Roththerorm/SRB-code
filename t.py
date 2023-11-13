import pandas as pd
from datetime import timedelta

print(6 % 5)

# with open('temperature_data.txt', 'r') as f:
#         data = f.read()
#         data_list = list(filter(None,data.replace(' ', '\t').split("\n")))
#         df = pd.DataFrame(data_list, columns=['single_data'])
#         df[['Day', 'Time', 'Temperature']] = df['single_data'].str.split('\t', expand=True)
#         df.drop('single_data', inplace=True, axis=1)
#         # df['Time'] = pd.to_datetime(df['Time']).dt.time
#         new_df = df.groupby(['Time'])['Temperature'].apply(lambda x: ''.join(x)).reset_index()

#         indexes = new_df[new_df['Temperature'].str.len() != 5].index
        
#         print(indexes)
#         valid_indexes = []

#         # Iterate through the specified indexes
#         for index in indexes:
#                 current_time = pd.to_datetime(new_df.at[index, 'Time'])
        
#                 if index > 0 and index < len(new_df) - 1:
#                         previous_time = pd.to_datetime(new_df.at[index - 1, 'Time'])
#                         next_time = pd.to_datetime(new_df.at[index + 1, 'Time'])
#                         if current_time - previous_time == timedelta(seconds=1) and next_time - current_time == timedelta(seconds=1):
#                                 value_temp = new_df.at[index, 'Temperature']
#                                 if len(value_temp) > 5:
#                                         tw
#                                 print(f"Index {index} is simultaneous.")

        




