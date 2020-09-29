#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def q_init(env):
    """[initializes the Q-table:]

    Args:
        env ([type]): [description]
    Returns: the Q-table as a numpy.ndarray of zeros
    """

    action_space_size = env.action_space.n
    states_space_size = env.observation_space.n

    q_table = np.zeros((states_space_size, action_space_size))

    return q_table
