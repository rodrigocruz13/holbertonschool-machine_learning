#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """[Function that uses epsilon-greedy to determine the next action:]

    Args:
        Q ([numpy.ndarray]): [array containing the q-table]
        state ([type]): [current state]
        epsilon ([type]): [epsilon to use for the calculation]
    """

    # https://www.youtube.com/watch?v=HGeI30uATws&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=9
    ε = epsilon
    # print(Q)
    # print(Q.shape)
    # print(Q[state])
    # print (state, epsilon)

    to_explore = np.random.randint(Q.shape[1])
    to_exploit = np.argmax(Q[state, :])
    exploration_rate_threshold = np.random.uniform(0, 1)

    movement = to_explore if ε > exploration_rate_threshold else to_exploit

    return movement

