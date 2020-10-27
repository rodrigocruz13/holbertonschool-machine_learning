#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


# Rename the column Timestamp to Date
df.rename(columns={"Timestamp": "Date"}, inplace=True)

# Convert the timestamp values to date values
df["Date"] = pd.to_datetime(df["Date"], unit="s")

# Index the data frame on Date
df = df.set_index("Date")

# The column Weighted_Price should be removed
df.drop(["Weighted_Price"], axis="columns", inplace=True)

# 2. missing values in High, Low, Open, and Close should be set to the previous
# row’s Close value
cols_na = ['High', 'Low', 'Open', 'Close']
for col in cols_na:
    df[col].fillna(method='ffill', inplace=True)

# 3. missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
cols_zero = ['Volume_(BTC)', 'Volume_(Currency)']
for col in cols_zero:
    df[col].fillna(0, inplace=True)


# Plot the data from 2017 and beyond at daily intervals
init_date = '2017-01-01'
df = df[(df.index > init_date)]
df = df[::1440]  # start, stop, step.  One single data per day

df.plot()
plt.show()
