import pandas as pd

def get_max_values(df):
    # idxmax() function to get the index of the row with the maximum value for each column
    max_rows = df.iloc[:, 1:].idxmax()

    # Initialize an empty list to store the tuples
    max_values = []

    # Iterate over the indexes of the rows with the maximum values
    for column, index in max_rows.items():
        # Append a tuple to the list with the value from the first column and the maximum value for the column
        max_values.append((df.loc[index, df.columns[0]], df.loc[index, column]))

    # Return the list of tuples
    return max_values
