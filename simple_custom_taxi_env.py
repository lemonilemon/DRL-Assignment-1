import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
from q_learning_agent import QLearningAgent, QNetwork
import random
# This environment allows you to verify whether your program runs correctly during testing,
# as it follows the same observation format from `env.reset()` and `env.step()`.
# However, keep in mind that this is just a simplified environment.
# The full specifications for the real testing environment can be found in the provided spec.
#
# You are free to modify this file to better match the real environment and train your own agent.
# Good luck!


class SimpleTaxiEnv:
    def __init__(self, grid_size=5, fuel_limit=50):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False

        self.passenger_loc = None
        # for y in range(0, self.grid_size):
        #     for x in range(0, self.grid_size):
        #         if random.random() < 0.3:
        #             self.obstacles.add((x, y))

        self.obstacles = set()
        for y in range(0, self.grid_size):
            for x in range(0, self.grid_size):
                if random.random() < 0.2:
                    self.obstacles.add((x, y))
        # self.obstacles = {
        #     (0, 1),
        #     (1, 1),
        #     (1, self.grid_size - 2),
        #     (1, self.grid_size - 1),
        #     (self.grid_size - 2, 0),
        #     (self.grid_size - 2, 1),
        #     (self.grid_size - 2, self.grid_size - 2),
        #     (self.grid_size - 1, self.grid_size - 2),
        # }  # No obstacles in simple version

        def select_non_adjacent_stations(available_stations, num_stations):
            selected_stations = []
            while len(selected_stations) < num_stations:
                station = random.choice(available_stations)
                if all(
                    abs(station[0] - s[0]) + abs(station[1] - s[1]) > 1
                    for s in selected_stations
                ):
                    selected_stations.append(station)
            return selected_stations

        available_stations = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.obstacles
        ]

        self.stations = select_non_adjacent_stations(available_stations, 4)
        self.destination = None

    def reset(self):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False

        self.obstacles = set()
        for y in range(0, self.grid_size):
            for x in range(0, self.grid_size):
                if random.random() < 0.2:
                    self.obstacles.add((x, y))

        def select_non_adjacent_stations(available_stations, num_stations):
            selected_stations = []
            while len(selected_stations) < num_stations:
                station = random.choice(available_stations)
                if all(
                    abs(station[0] - s[0]) + abs(station[1] - s[1]) > 1
                    for s in selected_stations
                ):
                    selected_stations.append(station)
            return selected_stations

        available_stations = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.obstacles
        ]

        self.stations = select_non_adjacent_stations(available_stations, 4)

        available_positions = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.obstacles and (x, y) not in self.stations
        ]

        self.taxi_pos = random.choice(available_positions)

        self.passenger_loc = random.choice([pos for pos in self.stations])

        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)

        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1

        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (
                0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size
            ):
                reward -= 5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc and not self.passenger_picked_up:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                    reward += 500
                else:
                    reward = -10
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 500
                        return self.get_state(), reward - 0.1, True, {}
                    else:
                        reward -= 1000
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -= 10

        reward -= 0.1

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward - 10, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination

        obstacle_north = int(
            taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles
        )
        obstacle_south = int(
            taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles
        )
        obstacle_east = int(
            taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles
        )
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = (
            passenger_loc_north
            or passenger_loc_south
            or passenger_loc_east
            or passenger_loc_west
            or passenger_loc_middle
        )

        destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east = int((taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west = int((taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle = int((taxi_row, taxi_col) == self.destination)
        destination_look = (
            destination_loc_north
            or destination_loc_south
            or destination_loc_east
            or destination_loc_west
            or destination_loc_middle
        )

        state = (
            taxi_row,
            taxi_col,
            self.stations[0][0],
            self.stations[0][1],
            self.stations[1][0],
            self.stations[1][1],
            self.stations[2][0],
            self.stations[2][1],
            self.stations[3][0],
            self.stations[3][1],
            obstacle_north,
            obstacle_south,
            obstacle_east,
            obstacle_west,
            passenger_look,
            destination_look,
        )
        return state

    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [["."] * self.grid_size for _ in range(self.grid_size)]

        for id, (sy, sx) in enumerate(self.stations, start=0):
            grid[sy][sx] = str(id)

        for oy, ox in self.obstacles:
            grid[oy][ox] = "#"
        # Place passenger
        py, px = self.passenger_loc
        # if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
        #     grid[py][px] = "P"
        # Place destination
        dy, dx = self.destination
        # if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
        #     grid[dy][dx] = "D"
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = "🚖"

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        print(f"Passenger Position: ({px}, {py})")
        print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = [
            "Move South",
            "Move North",
            "Move East",
            "Move West",
            "Pick Up",
            "Drop Off",
        ]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [
        (0, 0),
        (0, env.grid_size - 1),
        (env.grid_size - 1, 0),
        (env.grid_size - 1, env.grid_size - 1),
    ]

    (
        taxi_row,
        taxi_col,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look,
    ) = obs

    if render:
        env.render_env(
            (taxi_row, taxi_col), action=None, step=step_count, fuel=env.current_fuel
        )
        time.sleep(0.5)
    while not done:
        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        print("obs=", obs)
        total_reward += reward
        step_count += 1

        (
            taxi_row,
            taxi_col,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            obstacle_north,
            obstacle_south,
            obstacle_east,
            obstacle_west,
            passenger_look,
            destination_look,
        ) = obs

        if render:
            env.render_env(
                (taxi_row, taxi_col),
                action=action,
                step=step_count,
                fuel=env.current_fuel,
            )

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward


if __name__ == "__main__":
    env_config = {"grid_size": 10, "fuel_limit": 500}

    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")
