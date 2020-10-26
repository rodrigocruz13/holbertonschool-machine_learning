#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.iloc[::60, 2:6]  # start, stop, step, col_ini: col_end

print(df.tail())
