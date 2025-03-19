# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

fuel = 5000
have_passenger = False
# -1: not visted, 0: nothing, 1: passenger, 2: destination
station_check = [
    -1,
    -1,
    -1,
    -1,
]


def get_key(obs):
    global fuel, have_passenger, station_check
    stations = [[0, 0], [0, 0], [0, 0], [0, 0]]
    (
        taxi_row,
        taxi_col,
        stations[0][0],
        stations[0][1],
        stations[1][0],
        stations[1][1],
        stations[2][0],
        stations[2][1],
        stations[3][0],
        stations[3][1],
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look,
    ) = obs
    stations[0][0] -= taxi_row
    stations[0][1] -= taxi_col
    stations[1][0] -= taxi_row
    stations[1][1] -= taxi_col
    stations[2][0] -= taxi_row
    stations[2][1] -= taxi_col
    stations[3][0] -= taxi_row
    stations[3][1] -= taxi_col
    station_looks = [np.linalg.norm(stations[i], ord=1) <= 1 for i in range(4)]

    def sign(x):
        return 1 if x > 0 else -1 if x < 0 else 0

    #
    stations[0][0] = sign(stations[0][0])
    stations[0][1] = sign(stations[0][1])
    stations[1][0] = sign(stations[1][0])
    stations[1][1] = sign(stations[1][1])
    stations[2][0] = sign(stations[2][0])
    stations[2][1] = sign(stations[2][1])
    stations[3][0] = sign(stations[3][0])
    stations[3][1] = sign(stations[3][1])
    # size becomes 3 for each coordinate

    return (
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
    )
    # return (
    #     stations[0][0],  # 3
    #     stations[0][1],  # 3
    #     stations[1][0],  # 3
    #     stations[1][1],  # 3
    #     stations[2][0],  # 3
    #     stations[2][1],  # 3
    #     stations[3][0],  # 3
    #     stations[3][1],  # 3
    #     station_looks[0],  # 2
    #     station_looks[1],  # 2
    #     station_looks[2],  # 2
    #     station_looks[3],  # 2
    #     station_check[0],  # 4
    #     station_check[1],  # 4
    #     station_check[2],  # 4
    #     station_check[3],  # 4
    #     obstacle_north,  # 2
    #     obstacle_south,  # 2
    #     obstacle_east,  # 2
    #     obstacle_west,  # 2
    #     passenger_look,  # 2
    #     destination_look,  # 2
    #     have_passenger,  # 2
    # )


def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    stations = [[0, 0], [0, 0], [0, 0], [0, 0]]
    (
        taxi_row,
        taxi_col,
        stations[0][0],
        stations[0][1],
        stations[1][0],
        stations[1][1],
        stations[2][0],
        stations[2][1],
        stations[3][0],
        stations[3][1],
        _,
        _,
        _,
        _,
        passenger_look,
        destination_look,
    ) = obs
    global have_passenger, fuel, station_check
    for i in range(4):
        if taxi_row == stations[i][0] and taxi_col == stations[i][1]:
            if passenger_look:
                station_check[i] = 1
            elif destination_look:
                station_check[i] = 2
            else:
                station_check[i] = 0

    key = get_key(obs)
    with open("simple_q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    act = np.argmax(q_table[key])

    if act == 4 and passenger_look:  # If available pick up
        have_passenger = True
    elif act == 5 and have_passenger:  # If available drop off
        have_passenger = False
    if act == 5 and destination_look:  # If available drop off -> finish
        fuel = 5000
        have_passenger = False
        station_check = [-1] * 4

    fuel -= 1  # Consume fuel

    if fuel == 0:  # If fuel is 0 -> finish
        fuel = 0
        have_passenger = False
        station_check = [-1] * 4

    return act  # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
