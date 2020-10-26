#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
new_col_names = {'Timestamp': 'Datetime', }

# 1. Rename the column Timestamp to Datetime
df.rename(columns=new_col_names, inplace=True)

# 2 Convert the timestamp values to datatime values
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# 3. Display only the Datetime and Close columns

for col in df.columns:
    if (col != 'Datetime') and (col != 'Close'):
        df.drop(col, inplace=True, axis=1)

print(df.tail())
