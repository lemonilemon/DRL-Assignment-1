import numpy as np
import random
import pickle
from tqdm import tqdm
from utils import Get_Key
import env
import os
import gym
from simple_custom_taxi_env import SimpleTaxiEnv


def softmax(x):
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1)


class PolicyTableAgent:
    def __init__(
        self,
        env,
        learning_rate=1e-5,
        discount_factor=0.8,
    ):
        self.get_key = Get_Key()
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.policy_table = np.zeros(
            (
                3,
                3,
                2,
                2,
                2,
                2,
                2,
                2,
                6,
            ),
            dtype=np.float32,
        )

    def choose_action(self, key):
        action_probs = softmax(self.policy_table[key])
        return np.random.choice(6, p=action_probs)

    def learn(self, trajectory):
        G = 0
        returns = []
        for t in reversed(range(len(trajectory))):
            key, action, reward = trajectory[t]
            G = self.discount_factor * G + reward
            returns.append(G)
        returns.reverse()
        mean_return = np.mean(returns)
        std_return = np.std(returns) if np.std(returns) > 0 else 1
        for t in range(len(trajectory)):
            key, action, _ = trajectory[t]
            normalized_G = (returns[t] - mean_return) / std_return
            action_prob = softmax(self.policy_table[key])[action]
            self.policy_table[key][action] += (
                self.learning_rate * normalized_G * (1 - action_prob)
            )

    def train(self, episodes=200):
        total_reward_list = []
        steps_list = []
        trajectory = []
        env_config = {"grid_size": random.randint(5, 15), "fuel_limit": 5000}
        self.env = SimpleTaxiEnv(**env_config)
        for episode in tqdm(range(episodes)):
            self.get_key = Get_Key()
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                self.get_key.update_obs(state)
                key = self.get_key(state)
                action = self.choose_action(key)
                self.get_key.update_act(action)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((key, action, reward))
                state = next_state
                total_reward += reward
                steps += 1

            self.learn(trajectory)

            total_reward_list.append(total_reward)
            steps_list.append(steps)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(total_reward_list[-10:])
                avg_steps = np.mean(steps_list[-10:])
                print(
                    f"Episode {episode + 1}: Average Reward: {avg_reward}, Average Steps: {avg_steps}",
                    flush=True,
                )


if __name__ == "__main__":
    env_config = {"grid_size": 7, "fuel_limit": 5000}
    env = SimpleTaxiEnv(**env_config)
    agent = PolicyTableAgent(env)
    if os.path.exists("simple_policy_table.pkl"):
        agent.policy_table = pickle.load(open("simple_policy_table.pkl", "rb"))
    else:  # Set the initial policy
        agent.policy_table[:, :, :, 1, :, :, :, :, 0] = -20
        agent.policy_table[:, :, 1, :, :, :, :, :, 1] = -20
        agent.policy_table[:, :, :, :, 1, :, :, :, 2] = -20
        agent.policy_table[:, :, :, :, :, 1, :, :, 3] = -20
        agent.policy_table[:, :, :, :, :, :, 1, :, 4] = 20
        agent.policy_table[:, :, :, :, :, :, :, 1, 5] = 20
        agent.policy_table[:, :, :, :, :, :, 0, :, 4] = -20
        agent.policy_table[:, :, :, :, :, :, :, 0, 5] = -20
    agent.train(episodes=200)
    # Save the trained agent
    pickle.dump(agent.policy_table, open("simple_policy_table.pkl", "wb"))
