#!/usr/bin/env python3

build_model = __import__('0-sequential').build_model


if __name__ == '__main__':
    network = build_model(784,  # features
                          [256, 256, 10],  # layers
                          ['tanh', 'tanh', 'softmax'],  # activations
                          0.001,  # lambtha
                          0.95)  # keep_prob
    network.summary()
    print(network.losses)  # fdflsddsd
