#!/usr/bin/env python3

"""
Script that creates, trains & validates a keras model for the forecasting
of BTC:
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


def clean_screen():
    """
    Function that just clean the screen
    Arguments:  None
    Returns: Nothing
    """
    from os import system, name
    _ = system('cls') if name == 'nt' else system('clear')


def extract_fromzip(a_zipfile):
    """
    Function that extracts the file from a compressed zip archive
    Arguments: a_zipfile: a valid zip file
    Returns: None if fails or the decompressed file
    """
    import zipfile

    zf = zipfile.ZipFile(a_zipfile)
    try:
        print("Extracting from {} file".format(a_zipfile))
        filename = a_zipfile.split('.zip', 1)
        decompressed_file = zf.extract(filename[0])
    except BaseException as e:
        print(e)
        return None
    return decompressed_file


def read_csv(a_csv_file):
    """
    Function that read a csv file into a panda class
    Arguments: a_csv_file: a valid csv_file
    Returns: a pandas class dataframe
    """
    print("1. Opening {} file".format(a_csv_file))
    return pd.read_csv("./" + a_csv_file)


def p_process(full_df):
    """
    Function that preprocess some data from a pandas dataframe
    Arguments: full_df: a valid dataframe
    Returns: a smaller version of a_df with prepocessed data
    """

    # https://towardsdatascience.com/getting-started-with-bitcoin-historical-data-set-with-python-and-pandas-cd31417d1736

    # 2. Preprocessing the datasets
    print("2. Preprocessing data")

    # print("2.a. Calculating Closing variations")
    # stamps["Deltas"] = stamps["Close"].diff()

    print("2.b. Converting timestamps from UNIX to UTC")
    feature = 'Timestamp'
    full_df[feature] = pd.to_datetime(full_df[feature], unit='s')

    print("2.c. Cleaning data")
    print("     No data older than 2017")
    full_df["year"] = pd.DatetimeIndex(full_df["Timestamp"]).year
    full_df = full_df[full_df["year"] >= 2017]
    full_df = full_df.iloc[1:]  # removing first row
    full_df.drop(["year"], axis=1, inplace=True)

    print("     Interpolating no valid data")
    full_df = full_df.interpolate(method='linear')

    print("     No 'Open' column")
    full_df.drop(["Open"], axis=1, inplace=True)

    print("     No 'Volume_(Currency)' column")
    full_df.drop(["Volume_(Currency)"], axis=1, inplace=True)

    print("     No 'High' column")
    full_df.drop(["High"], axis=1, inplace=True)

    print("     No 'Low' column")
    full_df.drop(["Low"], axis=1, inplace=True)

    return full_df


def plotting_df(a_dataframe, feature):
    """
    Function that plots a feature from a dataframe file
    Arguments: a_csv_file: a valid csv_file
    Returns: a smaller version of the csv file with prepocessed data
    """
    print("3. Plotting data")
    """
    """


def saving_csv(a_dataframe):
    """
    Function that keeps the original file and save all changes in new location
    Arguments: a_dataframe: a valid dataframe
    Returns: None if fails
    """

    newfile_name = "current_btc_stamps.csv"
    print("4. Saving to a {}".format(newfile_name))
    a_dataframe.reset_index(inplace=True, drop=True)
    new_file = "./" + "current_btc_stamps.csv"
    try:
        a_dataframe.to_csv(new_file, index=False)
    except BaseException as e:
        print(e)
        return None


coin_base_csv_file = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
bc_stamps_csv_file = "bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"

# ** Main function **
clean_screen()
full_df = read_csv(bc_stamps_csv_file)
preprocesed_df = p_process(full_df)
saving_csv(preprocesed_df)
