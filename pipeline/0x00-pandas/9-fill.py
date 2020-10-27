#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# 1. The column Weighted_Price should be removed
df.drop("Weighted_Price", inplace=True, axis=1)

# 2. missing values in High, Low, Open, and Close should be set to the previous
# row’s Close value
cols_na = ['High', 'Low', 'Open', 'Close']
for col in cols_na:
    df[col].fillna(method='ffill', inplace=True)

# 3. missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
cols_zero = ['Volume_(BTC)', 'Volume_(Currency)']
for col in cols_zero:
    df[col].fillna(0, inplace=True)

print(df.head())
print(df.tail())
