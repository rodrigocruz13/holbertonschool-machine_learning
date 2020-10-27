#!/usr/bin/env python3

"""
Script that creates a RNN model to predicts the value of BTC for the next hour
"""

from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import pandas as pd
import os
from sklearn import preprocessing  # pip install sklearn
from collections import deque
import random
import time


def clean_screen():
    """
    Function that just clean the screen
    Arguments:  None
    Returns: Nothing
    """
    from os import system, name
    _ = system('cls') if name == 'nt' else system('clear')
    print("Bitcoin (฿) forecasting")
    print("Part 2. Predict")
    print("---------------------")
    print()


def read_csv(a_csv_file):
    """
    Function that read a csv file into a panda class
    Arguments: a_csv_file: a valid csv_file
    Returns: a pandas class dataframe
    """
    size_MB = os.path.getsize(a_csv_file) / 1000000
    print("1. Loading Dataframe {:>30} - {} MB".format(a_csv_file, size_MB))

    df = pd.read_csv("./" + a_csv_file)
    """
    col = 'Timestamp'

    df[col] = pd.to_datetime(df[col].dt.strftime('%Y-%m-%d %H:%M:%S'))

    f = '%Y-%m-%d %H:%M:%S'
    # df[col] = df[datetime.strptime(col, f)]

    # df['Timestamp'] = df['Timestamp'].apply(str_to_date(df['Timestamp']))
    """
    t11, t12 = df.index.min(), df.index.max()
    d_mi, d_mf = df.at[t11, 'Timestamp'], df.at[t12, 'Timestamp']
    print(d_mi, type(d_mi))
    d_mi = datetime.strptime(d_mi, '%Y-%m-%d %H:%M:%S')
    print(d_mi, type(d_mi))
    """
    d_mf = datetime.strptime(d_mf, '%Y-%m-%d %H:%M:%S')
    print(d_mi, d_mf, type(d_mi), type(d_mf))
    d_mi, d_mf = datetime.fromtimestamp(d_mi), datetime.fromtimestamp(d_mf)
    """
    ms = df.shape
    print("\tShape\tInit Date \t\tEnd date")
    print("\t{}".format(ms))

    return df


def data_seq(df, WINDOW_PREDICTION_HOURS, NAME):
    """[summary]
    Args:
        df ([type]): [description]
        WINDOW_PREDICTION_HOURS ([type]): [description]
    """

    # ------  1. Creating sequences
    print("7. Creating sequences of data for: {}".format(NAME))
    sequencial_data = []

    # Populate with new data and pops out the old
    previous_time = deque(maxlen=WINDOW_PREDICTION_HOURS)
    for row in df.values:
        previous_time.append([n for n in row[:-1]])  # all but Target column

        if len(previous_time) == WINDOW_PREDICTION_HOURS:
            # append features and labels; Row -1 = Target
            sequencial_data.append([np.array(previous_time), row[-1]])
    random.shuffle(sequencial_data)

    # ------  3. Balancing the data (The model will learn equally from ↑, ↓)
    print("\t7.a. Balancing both Classes: Ups (1) & Downs (0)")

    buys, sells = [], []
    for seq, target in sequencial_data:
        ls = sells if target == 0 else buys
        ls.append([seq, target])
    random.shuffle(buys), random.shuffle(sells)

    lower = min(len(buys), len(sells))
    buys, sells = buys[:lower], sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    print("\t7.b. Generating Sequential data and labels")
    X, y = [], []
    for seq, target in sequential_data:
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels
    return np.array(X), y


def verify_data_balance(train_X, train_Y, valid_X, valid_Y):
    """[summary]

    Args:
        train_X ([type]): [description]
        train_Y ([type]): [description]
        valid_X ([type]): [description]
        valid_Y ([type]): [description]
    """

    lt, lv = len(train_X), len(valid_Y)
    ty0, ty1 = train_Y.count(0), train_Y.count(1)
    vy0, vy1 = valid_Y.count(0), valid_Y.count(1)

    print("8. Class balance validation")
    print("\tTotal Train: {}\tClass 1: {}\tClass 0: {}".format(lt, ty0, ty1))
    print("\tTotal Valid: {}\tClass 1: {}\tClass 0: {}".format(lv, vy0, vy1))


def build_RNN(train_X):
    """[summary]
    """
    print("9. Building model")

    shape = (train_X.shape[1:])

    model = Sequential()
    model.add(LSTM(128, input_shape=shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print("   Model summary:")
    model.summary()

    return model


def train_model(model, train_X, train_y, valid_X, valid_y, FPH, WPH, E, B):
    """[summary]

    Args:
        model ([type]): [description]
        train_X ([type]): [description]
        train_y ([type]): [description]
        valid_X ([type]): [description]
        valid_y ([type]): [description]
    """
    print("10. Training RNN model")
    print("\tCreating Checkpoints")

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H.%M.%S")
    T = str(FPH) + "PRED" + date_time
    NAME = "BTC" + str(WPH) + T

    tensorboard = TensorBoard(log_dir="logs/{}".format("NAME"))
    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
    # saves only the best ones
    checkpoint = ModelCheckpoint("{}.model".format(filepath,
                                                   monitor='val_acc',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   mode='max'))
    # Train model
    history = model.fit(train_X,
                        train_y,
                        batch_size=B,
                        epochs=E,
                        validation_data=(valid_X, valid_y),
                        callbacks=[tensorboard, checkpoint])

    # Run evaluations: Score model
    score = model.evaluate(valid_X, valid_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save model
    model.save("{}".format(NAME))


def plotting_df(a_dataframe):
    """
    Function that plots a feature from a dataframe file
    Arguments: a_dataframe: a valid pandas dataframe
               feature: list of features to be plotted
    Returns: a smaller version of the csv file with prepocessed data
    """

    import matplotlib.pyplot as plt

    print("13. Plotting data")
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


def saving_csv(a_dataframe, name):
    """[Function that saves a dataframe into a file of name 'new_name']

    Args:
        a_dataframe ([pandas df]): [a valid pandas dataframe]
        new_name ([str]): [name of the file]

    Returns:
        [type]: [None if fails]
    """

    a_dataframe.reset_index(inplace=True, drop=True)
    new_file = "./" + name
    try:
        a_dataframe.to_csv(new_file, index=False)
        MB = os.path.getsize(name) / 1000000
        print("7. Saving to a file:\t'{:>12}'  Size = {} MB".format(name, MB))
        print()
    except BaseException as e:
        print(e)
        return None
