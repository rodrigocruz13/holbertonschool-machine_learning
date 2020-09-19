# 0x0E. Time Series Forecasting - BTC - ฿

## Resources
Read or watch:

Time Series Prediction
Time Series Forecasting
Time Series Talk : Stationarity
Tensorflow Datasets
Time Series Windowing
Definitions to skim

Time Series
Stationary Process
References:

tf.keras.layers.SimpleRNN
tf.keras.layers.GRU
tf.keras.layers.LSTM
tf.data.Dataset

Tasks

0. When to Invest mandatory
Bitcoin (BTC) became a trending topic after its price peaked in 2018. Many
have sought to predict its value in order to accrue wealth. Let’s attempt to
use our knowledge of RNNs to attempt just that.

Given the [coinbase](https://intranet.hbtn.io/rltoken/_-9LQxYpc6qTM7K_AI58-g)
and [bitstamp](https://intranet.hbtn.io/rltoken/0zZKYc5-xlxGFbxTfCVrBA)
datasets, write a script, forecast_btc.py, that creates, trains, and validates
a keras model for the forecasting of BTC:

Your model should use the past 24 hours of BTC data to predict the value of
BTC at the close of the following hour (approximately how long the average
transaction takes):
The datasets are formatted such that such that every row represents a 60 sec
time window containing:
- The start time of the time window in Unix time
- The open price in USD at the start of the time window
- The high price in USD within the time window
- The low price in USD within the time window
- The close price in USD at end of the time window
- The amount of BTC transacted in the time window
- The amount of Currency (USD) transacted in the time window
- The volume-weighted average price in USD for the time window

Your model should use an RNN architecture of your choosing
Your model should use mean-squared error (MSE) as its cost function
You should use a tf.data.Dataset to feed data to your model

Because the dataset is raw, you will need to create a script named
preprocess_data.py to preprocess this data. Here are some things to consider:

- Are all of the data points useful?
- Are all of the data features useful?
- Should you rescale the data?
- Is the current time window relevant?
- How should you save this preprocessed data?

## Files
- README.md
- forecast_btc.py,
- preprocess_data.py


## General requirements
* Allowed editors: vi, vim, emacs
* All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
* Your files will be executed with numpy (version 1.15) and tensorflow (version 1.12)
* All your files should end with a new line
* The first line of all your files should be exactly #!/usr/bin/env python3
* All of your files must be executable
* A README.md file, at the root of the folder of the project, is mandatory
* Your code should follow the pycodestyle style (version 2.4)
* All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
* All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
* All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')

## Installation
In your terminal, git clone the directory with the following command:
```
git clone https://github.com/rodrigocruz13/holbertonschool-interview_prep
cd 0x05-menger
```

Compile the files using:

```
gcc -Wall -Wextra -Werror -pedantic -o 0-menger -g 0-menger.c 0-main.c -lm
```

## Usage

Run the program using

```
./preprocess.py
./forecast_btc.py
```
