# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import torch
import random
from utils import Get_Key


get_key = Get_Key()  # Create a get_key functor


def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    get_key.update_obs(obs)
    key = get_key(obs)
    policy_table = pickle.load(open("simple_policy_table.pkl", "rb"))

    def softmax(x):
        x_max = np.max(x)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1)

    action_probs = softmax(policy_table[key])
    act = np.random.choice(6, p=action_probs)

    get_key.update_act(act)
    return act  # Choose a random action


if __name__ == "__main__":
    pass
