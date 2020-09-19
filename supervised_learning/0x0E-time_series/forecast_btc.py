#!/usr/bin/env python3

"""
Script that preprocess data for the forecasting the value of BTC:
"""

from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from sklearn import preprocessing  # pip install sklearn
from collections import deque
import random
import zipfile


def clean_screen():
    """
    Function that just clean the screen
    Arguments:  None
    Returns: Nothing
    """
    from os import system, name
    _ = system('cls') if name == 'nt' else system('clear')
    print("Bitcoin (฿) forecasting")
    print("Part 1. Preprocessing data")
    print("---------------------")
    print()


def extract_fromzip(a_zipfile):
    """
    Function that extracts the file from a compressed zip archive
    Arguments: a_zipfile: a valid zip file
    Returns: None if fails or the decompressed file
    """

    size_MB = os.path.getsize(a_zipfile) / 1000000
    zf = zipfile.ZipFile(a_zipfile)
    print("0. Decompressing {} ".format(a_zipfile), end="")
    print("\b Size = {} MB".format(size_MB))

    try:
        filename = a_zipfile.split('.zip', 1)
        decompressed_cvs_file = zf.extract(filename[0])
    except BaseException as e:
        print(e)
        return None
    return decompressed_cvs_file


def read_csv(a_csv_file):
    """
    Function that read a csv file into a panda class
    Arguments: a_csv_file: a valid csv_file
    Returns: a pandas class dataframe
    """
    size_MB = os.path.getsize(a_csv_file) / 1000000
    print("1. Opening {}".format(a_csv_file), end="\t\t")
    print("\b  Size = {} MB".format(size_MB))
    return pd.read_csv("./" + a_csv_file)


def select_df(df1, name1, df2, name2):
    """
    Function that selec the DF with the most updated info between 2 pandas DFs

    Args:
        df1 ([pandas DF]): [1st DF with BTC trade info]
        df2 ([pandas DF]): [2nd DF with BTC trade info]
        name1 ([str]): [Name of the 1st dataframe]
        name2 ([srt]): [Name of the 2nd dataframe]
    """

    print("2. Selecting most recent data be use used by our Dataframe")
    t1, t2 = len(df1.index) - 1, len(df2.index) - 1

    df1_last_date = datetime.fromtimestamp(df1.at[t1, 'Timestamp'])
    df2_last_date = datetime.fromtimestamp(df2.at[t2, 'Timestamp'])

    df, name = (df1, name1) if df1_last_date > df2_last_date else (df2, name2)

    print("\t{}. Rows = {}. Last date = {}.".format(name1, t1, df1_last_date))
    print("\t{}. Rows = {}. Last date = {}.".format(name2, t2, df2_last_date))

    print("\tDataframe selected: {}".format(name))

    sliced_df = df[::60] # This step should be at preprocessing data but
    return sliced_df  # it is used here to reduce the size of the Database.


def classify(current, future):
    """[Function that tells if the future value is bigger than current value]

    Args:
        current ([str]): [string describing the current value of BTC]
        future ([str]): [string describing the future value of BTC]

    Returns:
        [int]: [1 if future is > than current, 0 otherwise]
    """
    value = 1 if float(future) > float(current) else 0

    return value


def target(df, FUTURE_PREDICTION_HOURS):
    """[Function that generates the column Target in the dataset]

    Args:
        df ([pandas df]): [dataframe with the trading info of BTC]
        FUTURE_PREDICTION_HOURS ([int]): [hours to be predicted]

    Returns:
        df ([pandas df]): df with a new column added
    """

    print("3. Generating Target column")
    df['Future_USD'] = df['Close'].shift(-FUTURE_PREDICTION_HOURS)
    df['Target'] = list(map(classify, df['Close'], df['Future_USD']))
    df = df.drop(["Future_USD"], axis=1)

    return df


def split_validation_df(df, VALIDATION_STARTS_AT):
    """[Splits the original df in two dataframes: validation and df]

    Args:
        df ([pandas df]): [dataframe with the trading info of BTC]
        VALIDATION_STARTS_AT ([float]): [% where the validation Df starts]

    Returns:
        [validation_df]: [validation dataframe]
        [df]: [sliced df at 95 % of the original df]
    """

    print("4. Separating validation data from working data")

    print("\t Current size of main DF = {}", df.shape)
    print("\t Breaking DF at {} %".format(VALIDATION_STARTS_AT * 100))
    breaking_time = int((len(df.index) - 1) * VALIDATION_STARTS_AT)
    main_df, vali_df = df.iloc[: breaking_time], df.iloc[breaking_time: ]

    """
    ms, vs =main_df.shape, vali_df.shape
    t11, t12 = main_df.index.min(), main_df.index.max()
    t21, t22 = vali_df.index.min(), vali_df.index.max()

    d_ini1, d_end1 = main_df.at[t11, 'Timestamp'], main_df.at[t12, 'Timestamp']
    d_ini2, d_end2 = vali_df.at[t21, 'Timestamp'], vali_df.at[t22, 'Timestamp']
    """

    ms, vs = main_df.shape, vali_df.shape
    lm, lv = len(main_df), len(vali_df)
    print("\t Size of main DF is {}, with {} registers".format(ms, lm))
    print("\t Size of Validation DF = {}, with {} registers".format(vs, lv))
    return vali_df, main_df


def preprocess(full_df):
    """
    Function that preprocess some data from a pandas dataframe
    Arguments: full_df: a valid pandas dataframe
    Returns: a smaller version of a_df with prepocessed data
    """
    # https://bit.ly/3bM9VUl

    # 5. Preprocessing the dataset
    print("6. Preprocessing data")

    print("6.a. Converting Timestamps from UNIX to UTC")
    feature = 'Timestamp'
    full_df[feature] = pd.to_datetime(full_df[feature], unit='s')

    print("6.b. Filling the gaps. Interpolating NAN with last known value")
    full_df.fillna(method='pad', inplace=True)
    # print(full_df.head())
    # main_df.fillna(method="ffill", inplace=True)
    # print(full_df.head())

    print("6.c. Slicing data:")
    init_year = 2017

    print("\tDataframe current shape = {}".format(full_df.shape))
    print("\t\t- Removing data older than {}".format(init_year))
    full_df["year"] = pd.DatetimeIndex(full_df["Timestamp"]).year
    full_df = full_df[full_df["year"] >= init_year]
    sliced_df = full_df.drop(["year"], axis=1)

    print("\t\t- Subsamplig data to only use data each 60 min intervals")
    # sliced_df = sliced_df[::60]  # start, stop, step.
    # This step was done at # 2 but it is officially part of preprocessing.

    print("\t\t- Generating price variations (Δ price)")
    sliced_df["Variation"] = sliced_df["Close"].diff()

    print("\t\t- Removing 'Open' column")
    sliced_df.drop(["Open"], axis=1, inplace=True)

    print("\t\t- Removing 'High' column")
    sliced_df.drop(["High"], axis=1, inplace=True)

    print("\t\t- Removing 'Low' column")
    sliced_df.drop(["Low"], axis=1, inplace=True)

    print("\t\t- Removing 'Volume_(BTC)' column")
    sliced_df.drop(["Volume_(BTC)"], axis=1, inplace=True)

    print("\t\t- Removing 'Weighted_Price' column")
    sliced_df.drop(["Weighted_Price"], axis=1, inplace=True)

    new_col_names = {'Close': 'Close_USD','Volume_(Currency)': 'Vol_USD'}

    print("\t\t- Renaming remaining columns")
    sliced_df.rename(columns=new_col_names, inplace=True)
    print("\tDataframe current shape = {}".format(sliced_df.shape))

    print("6.d. Normalizing data: Converting to % and then into a range(0, 1)")

    # ------  1. Normalizing data
    for col in sliced_df.columns:
        if (col != "Target" and col != 'Timestamp' and col != 'Variation'):  # normalize all except the 'Target = Y' column

            # normalizes the column according to its % of change (pct_change)
            sliced_df[col] = sliced_df[col].pct_change()
            sliced_df.dropna(inplace=True)

            # scale the values from 0 to 1
            sliced_df[col] = preprocessing.scale(sliced_df[col].values)
    sliced_df.dropna(inplace=True)

    return sliced_df

    # ------  2. Creating sequences
    print("6.e. Creating sequences of data")
    sequencial_data = []

    # Populate with new data and pops out the old
    previous_time = deque(maxlen=WINDOW_PREDICTION_HOURS)
    for row in sliced_df.values:
        previous_time.append([n for n in row[:-1]])  # all but Target column

        if len(previous_time) == WINDOW_PREDICTION_HOURS:
            # append features and labels
            sequencial_data.append([np.array(previous_time), row[-1]])
    random.shuffle(sequencial_data)

    # ------  3. Balancing the data (The model will learn equally from ↑, ↓)
    print("6.f. Balancing data: Have the same number of classes: Ups & Downs")
    buys, sells = [], []
    for seq, target in sequencial_data:
        ls = sells if target == 0 else buys
        ls.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    # what's the shorter length?
    lower = min(len(buys), len(sells))

    # make sure both lists are only up to the shortest length.
    buys, sells = buys[:lower], sells[:lower]
    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X, y = [], []
    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!


def plotting_df(a_dataframe):
    """
    Function that plots a feature from a dataframe file
    Arguments: a_dataframe: a valid pandas dataframe
               feature: list of features to be plotted
    Returns: a smaller version of the csv file with prepocessed data
    """

    import matplotlib.pyplot as plt

    print("5. Plotting data")
    x_data = a_dataframe['Timestamp']
    try:
        y_data = a_dataframe['Close_USD']
    except:
        y_data = a_dataframe['Close']

    # 2. Form
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))

    # 3. font
    plt.rcParams.update({'font.size': 12})

    # 4.Labels
    plt.xlabel('Date')
    plt.ylabel('Closing value (US$)')
    plt.title("Bitcoin (BTC) - ฿")

    # 5. Display
    line_color = '#aec7e8'  # line color red
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(x_data, y_data, line_color, linewidth=1.45)
    plt.grid(True, ls='--', lw=.5, c='k', alpha=.13)
    plt.show()


def saving_csv(a_dataframe, new_name):
    """[Function that saves a dataframe into a file of name 'new_name']

    Args:
        a_dataframe ([pandas df]): [a valid pandas dataframe]
        new_name ([str]): [name of the file]

    Returns:
        [type]: [None if fails]
    """

    a_dataframe.reset_index(inplace=True, drop=True)
    new_file = "./" + new_name
    try:
        a_dataframe.to_csv(new_file, index=False)
        size_MB = os.path.getsize(new_name) / 1000000
        print("7. Saving to a file:\t'{}'  Size = {} MB".format(new_name, size_MB), end="\t\t\t")
        print()
    except BaseException as e:
        print(e)
        return None


if __name__ == "__main__":

    BITSTAMP_USD_CSV = "bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"
    COINBASE_USD_CSV = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"

    WINDOW_PREDICTION_HOURS = 24
    FUTURE_PREDICTION_HOURS = 1
    VALIDATION_STARTS_AT = 0.95  # 95% for data and last 5% to validation

    clean_screen()
    pd.set_option('mode.chained_assignment', None) # Avoid warnings

    # 0. Extracting CSV from zip files
    extract_fromzip(BITSTAMP_USD_CSV + '.zip')
    extract_fromzip(COINBASE_USD_CSV + '.zip')
    print()

    # 1. Opening CSV files and converting them into pandas dataframes
    df1, df2 = read_csv(BITSTAMP_USD_CSV), read_csv(COINBASE_USD_CSV)
    print()

    # 2. Selecting which one to use
    full_df = select_df(df1, "BITSTAMP_USD", df2, "COINBASE_USD")
    print()

    # 3. Generate Target column (Y)
    df = target(full_df, FUTURE_PREDICTION_HOURS)
    print()

    # 4. Separate validation_dataframe from the rest of the data
    validation_df, df = split_validation_df(df, VALIDATION_STARTS_AT)
    # saving_csv(validation_df, "Validation.csv")
    print()

    # 5. Plotting
    plotting_df(df)
    # print()

    # 6. Preprocessing the selected Dataframe (Slicing data)
    df = preprocess(df)
    print()

    # 7. Saving data
    saving_csv(df, "BTC_trade_info.csv")
    saving_csv(validation_df, "validation_BTC_trade_info.csv")
