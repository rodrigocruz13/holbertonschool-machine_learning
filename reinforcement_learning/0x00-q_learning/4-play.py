#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def play(env, Q, max_steps=100):
    """[Function that has the trained agent play an episode]

    Args:
        env ([instance]):   [FrozenLakeEnv instance]
        Q ([ndarray]):      [is a numpy.ndarray containing the Q-table]
        max_steps (int, optional): [maximum number of steps in the episode].
                                    Defaults to 100.

    Each state of the board should be displayed via the console
    You should always exploit the Q-table

    Returns: the total rewards for the episode
    """
    # Reset the enviroment
    state = env.reset()

    # Finished game ?
    done = False
    env.render()
    for _ in range(max_steps):
        call_to_action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(call_to_action)
        env.render()

        if (reward and done):
            print(reward)
            break
        state = new_state
    env.close()
