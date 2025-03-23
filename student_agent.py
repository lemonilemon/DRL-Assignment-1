# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import torch
import random
from utils import Get_Key
# from q_learning_agent import QNetwork
# from policy_agent import PolicyNetwork


get_key = Get_Key()  # Create a get_key functor


def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    q_table = pickle.load(open("simple_q_table.pkl", "rb"))
    get_key.update_obs(obs)
    key = get_key(obs)
    act = np.argmax(q_table[key])
    # print(
    #     f"Key = {get_key(obs)}, cur_dest = {get_key.cur_destination_station}, fuel = {get_key.fuel}, weights = {q_table[key]}"
    # )
    get_key.update_act(act)
    # model = PolicyNetwork(8, 6)
    # model.load_state_dict(torch.load("policy_network.pth", weights_only=False))
    # model.eval()
    # key = get_key(obs)
    # with torch.no_grad():
    #     probs_weights = model(torch.tensor(key, dtype=torch.float32))
    #     act = torch.argmax(probs_weights).item()
    return act  # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.


if __name__ == "__main__":
    model = PolicyNetwork(8, 6)
    model.load_state_dict(torch.load("policy_network.pth", weights_only=False))
    model.eval()
    q_table = pickle.load(open("simple_q_table.pkl", "rb"))
    print(q_table)
    # for i in range(2**8):
    #     binary_value = format(i, f"0{8}b")
    #     key = tuple([int(bit) for bit in binary_value])
    #     print(f"Key: {key}, Q-value: {q_table[key]}")
    # with torch.no_grad():
    #     probs_weights = model(torch.tensor(key, dtype=torch.float32))
    #     act = torch.argmax(probs_weights).item()
    #     print(f"Key: {key}, probs_weights: {probs_weights}, Action: {act}")
