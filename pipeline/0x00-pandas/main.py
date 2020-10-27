#!/usr/bin/env python3

import numpy as np
import pandas as pd


clean_sc = __import__('preprocess_data').clean_screen
open_zip = __import__('preprocess_data').extract_fromzip
read_csv = __import__('preprocess_data').read_csv
select_d = __import__('preprocess_data').select_df
target__ = __import__('preprocess_data').target
split_df = __import__('preprocess_data').split_validation_df
plotting = __import__('preprocess_data').plotting_df
preproc_ = __import__('preprocess_data').preprocess

data_seq = __import__('forecast_btc').data_seq
balances = __import__('forecast_btc').verify_data_balance
RNN_make = __import__('forecast_btc').build_RNN
train_md = __import__('forecast_btc').train_model

pd.set_option('mode.chained_assignment', None)  # Avoid warnings

BITSTAMP_USD_CSV = "bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
COINBASE_USD_CSV = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"

WPH = WINDOW_PREDICTION_HOURS = 24
FPH = FUTURE_PREDICTION_HOURS = 1

VSA = VALIDATION_STARTS_AT = 0.95  # 95% for data and last 5% to validation

E = EPOCHS = 15
B = BATCH_SIZE = 64

clean_sc()

print("Bitcoin (฿) forecasting")
print("Part 1. Preprocessing data")
print("---------------------")
print()

# 0. Extracting CSV from zip files
open_zip(BITSTAMP_USD_CSV + '.zip')
open_zip(COINBASE_USD_CSV + '.zip')
print()

# 1. Opening CSV files and converting them into pandas dataframes
df1, df2 = read_csv(BITSTAMP_USD_CSV), read_csv(COINBASE_USD_CSV)
print()

# 2. Selecting which one to use
full_df = select_d(df1, "BITSTAMP_USD", df2, "COINBASE_USD")
print()

# 3. Generate Target column (Y)
df = target__(full_df, FPH)
print()

# 4. Separate validation_dataframe from the rest of the data
validation_df, df = split_df(df, VSA)
# saving_csv(validation_df, "Validation.csv")
print()

"""
# 5. Plotting
plotting(df)
print()
"""

# 6. Preprocessing the selected Dataframe (Slicing data)
main_df = preproc_(df, "Main DF")
print()
vali_df = preproc_(validation_df, "Validation DF")
print()

print("Bitcoin (฿) forecasting")
print("Part 2. Predicting values")
print("---------------------")
print()

# 7. Generate data sequences
train_X, train_Y = data_seq(main_df, WPH, "Training")
print()
valid_X, valid_Y = data_seq(vali_df, WPH, "Validation")
print()

# 8. Confirming sizes
balances(train_X, train_Y, valid_X, valid_Y)
print()

# 9. Building and compile the model
model = RNN_make(train_X)
print()

# 10. Train the model
train_md(model, train_X, train_Y, valid_X, valid_Y, FPH, WPH, E, B)
print()

"""
# 11 Plotting new results

# 12. Saving data
# saving_csv(main_df, "main_DF.csv")
# saving_csv(vali_df, "validation_DF.csv")
"""
