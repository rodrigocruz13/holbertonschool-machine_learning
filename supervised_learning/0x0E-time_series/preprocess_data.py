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
    [Function that just clean the screen]

    Args:
        None

    Returns:
        Nothing
    """

    from os import system, name
    _ = system('cls') if name == 'nt' else system('clear')


def extract_fromzip(a_zipfile):
    """
    [Function that extracts a file from a zip archive]

    Args:
        a_zipfile ([type]): [a valid zip file]

    Returns:
        None if fails or the decompressed file
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
    [Function that read a csv file can convert the data into a panda Df]

    Args:
        a_csv_file ([file]):        [CSV file with BTC trading info]

    Returns:
        df         ([Pandas DF]):   [A Pandas dataframe]
    """

    size_MB = os.path.getsize(a_csv_file) / 1000000
    print("1. Opening {}".format(a_csv_file), end="\t\t")
    print("\b  Size = {} MB".format(size_MB))
    df = pd.read_csv("./" + a_csv_file, error_bad_lines=False)

    return df


def select_df(df1, name1, df2, name2):
    """
    [Function that selects the most updated DF]

    Args:
        df1     ([Pandas DF]):  [1st DF with BTC trade info]
        df2     ([Pandas DF]):  [2nd DF with BTC trade info]
        name1   ([str]):        [Name of the 1st dataframe]
        name2   ([srt]):        [Name of the 2nd dataframe]

    Returns:
        df      ([Pandas DF]):   [The most updated Pandas dataframe]
    """

    print("2. Selecting most recent data to be used by our Dataframe")
    t1, t2 = len(df1.index) - 1, len(df2.index) - 1

    df1_last_date = datetime.fromtimestamp(df1.at[t1, 'Timestamp'])
    df2_last_date = datetime.fromtimestamp(df2.at[t2, 'Timestamp'])

    df, name = (df1, name1) if df1_last_date > df2_last_date else (df2, name2)

    print("\t{}. Rows = {}. Last date = {}.".format(name1, t1, df1_last_date))
    print("\t{}. Rows = {}. Last date = {}.".format(name2, t2, df2_last_date))

    print("\tDataframe selected: {}".format(name))

    return df


def classify(current, future):
    """
    [Function that tells evaluates if 'future' > 'current']

    Args:
        Current     ([float]):      [Current value of BTC]
        future      ([float]):      [Future value of BTC]

    Returns:
        value       ([int]):        [1 if 'future' > 'current', 0 otherwise]
    """
    value = 1 if float(future) > float(current) else 0

    return value


def target(df, FPH):
    """
    [Function that generates the column (class) Target in the dataset]

    Args:
        df  ([Pandas df]):  [dataframe with the trading info of BTC]
        FPH ([int]):        [hours to be predicted]

    Returns:
        df  ([pandas df]):  Df with a new column added
    """

    print("3. Generating Target column (Class Up = 1, Down = 0)")
    df['Future_USD'] = df['Close'].shift(-FPH)
    df['Target'] = list(map(classify, df['Close'], df['Future_USD']))
    df = df.drop(["Future_USD"], axis=1)

    return df


def split_validation_df(df, VSA):
    """
    [Splits the original Df in two dataframes: validation df and main df]

    Args:
        df      ([Pandas df]):  [Dataframe with the trading info of BTC]
        VSA     ([float]):      [% where the Validation (DF) Starts At]

    Returns:
        vali_df ([Pandas df]):  [Validation DF]
        main_df ([Pandas df]):  [main Df sliced at 95 % of the original DF]
    """

    print("4. Separating Main DF from Validation DF")

    pc = VSA * 100
    print("   Current DF shape: {}. Splitting it at {}%".format(df.shape, pc))
    breaking_time = int((len(df.index) - 1) * VSA)
    main_df, vali_df = df.iloc[: breaking_time], df.iloc[breaking_time:]

    t11, t12 = main_df.index.min(), main_df.index.max()
    t21, t22 = vali_df.index.min(), vali_df.index.max()
    d_mi, d_mf = main_df.at[t11, 'Timestamp'], main_df.at[t12, 'Timestamp']
    d_vi, d_vf = vali_df.at[t21, 'Timestamp'], vali_df.at[t22, 'Timestamp']
    d_mi, d_mf = datetime.fromtimestamp(d_mi), datetime.fromtimestamp(d_mf)
    d_vi, d_vf = datetime.fromtimestamp(d_vi), datetime.fromtimestamp(d_vf)

    ms, vs = main_df.shape, vali_df.shape
    print("\t Main:\tNew shape\tInit Date \t\tEnd date")
    print("\t\t{}\t{}\t{}".format(ms, d_mi, d_mf))
    print()
    print("\t Valid:\tNew shape\tInit Date \t\tEnd date")
    print("\t\t{}\t{}\t{}".format(vs, d_vi, d_vf))

    return vali_df, main_df


def preprocess(full_df, NAME):
    """
    [Function that preprocess data from a Pandas Df]

    Args:
        full_df ([Pandas df]):  [Dataframe with the trading info of BTC]
        NAME     ([str]):       [Name of the dataframe]

    Returns:
        main_df ([Pandas df]):  [A sliced preprocessed version of the DF]
    """

    print("6. Preprocessing data of {}".format(NAME))

    # a. Converting Timestamps
    print("6.a. Converting Timestamps from UNIX to UTC")
    full_df['Timestamp'] = pd.to_datetime(full_df['Timestamp'], unit='s')

    # b. Filling the GAPs
    print("6.b. Filling the gaps. Interpolating NAN with last known value")
    full_df.fillna(method='pad', inplace=True)
    # main_df.fillna(method="ffill", inplace=True)

    # c. Slicing data
    print("6.c. Slicing data:")
    init_year = 2017

    print("\tDataframe current shape = {}".format(full_df.shape))
    print("\t\t- Removing data older than {}".format(init_year))

    full_df["year"] = pd.DatetimeIndex(full_df["Timestamp"]).year
    full_df = full_df[full_df["year"] >= init_year]
    sliced_df = full_df.drop(["year"], axis=1)

    print("\t\t- Subsamplig data to only use data each 60 min intervals")
    sliced_df = sliced_df[::60]  # start, stop, step.

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

    new_col_names = {'Close': 'Close_USD', 'Volume_(Currency)': 'Vol_USD'}

    print("\t\t- Renaming remaining columns")
    sliced_df.rename(columns=new_col_names, inplace=True)

    print("\tDataframe current shape = {}".format(sliced_df.shape))

    # d. Normalizing data
    print("6.d. Normalizing data: Converting to percetages")

    for col in sliced_df.columns:
        # Normalizing all but 'Target = Y', and Timestamp columns
        if (col != "Target" and col != 'Timestamp'):
            # normalizes the column according to its % of change (pct_change)
            sliced_df[col] = sliced_df[col].pct_change()
            sliced_df.dropna(inplace=True)

    # e. Scaling data
    print("6.e. Scaling data: Converting to a range from 0 to 1")
    for col in sliced_df.columns:
        # Scaling the values from 0 to 1
        if (col != "Target" and col != 'Timestamp'):
            sliced_df[col] = preprocessing.scale(sliced_df[col].values)
    sliced_df.dropna(inplace=True)

    # Restoring the values of Timestamps
    """
    dates = pd.to_datetime(['2019-01-15 13:30:00'])
    (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    # Int64Index([1547559000], dtype='int64')
    """
    t = pd.Timestamp("1970-01-01")
    sliced_df['Timestamp'] = (sliced_df['Timestamp'] - t) / pd.Timedelta('1s')

    return sliced_df


def plotting_df(a_dataframe):
    """
    [Function that prints X and Y data from a Pandas DF]

    Args:
        a_dataframe ([pandas df]): [a valid pandas dataframe]

    Returns:
        Nothing
    """

    import matplotlib.pyplot as plt

    print("5. Plotting data")
    x_data = pd.to_datetime(a_dataframe['Timestamp'], unit='s')

    try:
        y_data = a_dataframe['Close_USD']
    except BaseException:
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


def saving_csv(df, new_name):
    """
    Function that saves a DF into a file of name 'new_name']

    Args:
        df ([pandas df]): [a valid pandas dataframe]
        new_name ([str]): [name of the file]

    Returns:
        None
    """

    # df.reset_index(inplace=True, drop=True)
    new_file = "./" + new_name
    try:
        df.to_csv(new_file, index=False)
        MB = os.path.getsize(new_name) / 1000000
        print("7. Saving file: '{:>17}'\tSize = {} MB".format(new_name, MB))

    except BaseException as e:
        print(e)

    return None
