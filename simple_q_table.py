import numpy as np
import random
import pickle
from tqdm import tqdm
from student_agent import get_key
from simple_custom_taxi_env import SimpleTaxiEnv


class QLearningAgent:
    def __init__(
        self,
        env,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.999,
        min_exploration_rate=0.01,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = {}

    def get_state_key(self, state):
        return get_key(state)

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(6))
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(6)
            return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(6)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(6)

        q_value = self.q_table[state_key][action]
        max_next_q_value = np.max(self.q_table[next_state_key])

        td_target = reward + self.discount_factor * max_next_q_value * (1 - done)
        td_error = td_target - q_value

        self.q_table[state_key][action] += self.learning_rate * td_error

    def train(self, episodes=300):
        total_reward_list = []
        steps_list = []
        for episode in tqdm(range(episodes)):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

            total_reward_list.append(total_reward)
            steps_list.append(steps)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(total_reward_list[-10:])
                avg_steps = np.mean(steps_list[-10:])
                print(
                    f"Episode {episode + 1}: Average Reward: {avg_reward}, Average Steps: {avg_steps}",
                    flush=True,
                )

            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay,
            )


if __name__ == "__main__":
    env_config = {"grid_size": 7, "fuel_limit": 5000}
    env = SimpleTaxiEnv(**env_config)
    agent = QLearningAgent(env)
    agent.train(episodes=3000)
    print(agent.q_table)
    # Save the trained agent
    pickle.dump(agent.q_table, open("simple_q_table.pkl", "wb"))
