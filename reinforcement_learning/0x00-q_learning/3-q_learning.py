#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
        env,
        Q,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1,
        min_epsilon=0.1,
        epsilon_decay=0.05):
    """[summary]

    Args:
        env ([type]): [description]
        Q ([type]): [description]
        episodes (int, optional): [description]. Defaults to 5000.
        max_steps (int, optional): [description]. Defaults to 100.
        alpha (float, optional): [description]. Defaults to 0.1.
        gamma (float, optional): [description]. Defaults to 0.99.
        epsilon (int, optional): [description]. Defaults to 1.
        min_epsilon (float, optional): [description]. Defaults to 0.1.
        epsilon_decay (float, optional): [description]. Defaults to 0.05.

    Returns:
        [type]: [description]
    """

    α = alpha
    ε = epsilon
    γ = gamma

    total_rewards = []

    for a_single_episode in range(episodes):
        state = env.reset()
        done = False
        # reward current episode

        rewards = 0
        for step in range(max_steps):

            call_to_action = epsilon_greedy(Q, state, ε)
            new_state, a_single_reward, done, info = env.step(call_to_action)

            old_value = Q[state, call_to_action]
            new_value = (a_single_reward + γ * np.max(Q[new_state, :]))
            Q[state, call_to_action] = (old_value * (1 - α)) + (new_value * α)

            state = new_state

            if done is True:
                if a_single_reward != 1:
                    rewards = -1
                rewards = rewards + a_single_reward
                break
            else:
                rewards = rewards + a_single_reward

        total_rewards.append(rewards)
        ε = min_epsilon + (1 - min_epsilon) * \
            np.exp(-epsilon_decay * a_single_episode)

    return Q, total_rewards
