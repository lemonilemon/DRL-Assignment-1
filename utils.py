import numpy as np


class Get_Key:
    def __init__(self):
        self.fuel = 5000
        self.have_passenger = False
        self.cur_destination_station = None
        # -1: not visted, 0: nothing, 1: passenger, 2: destination
        self.station_check = [-1] * 4
        self.in_station = None
        self.passenger_station = None
        self.destination_station = None
        self.last_action = 5

    def reset(self):
        self.fuel = 5000
        self.have_passenger = False
        self.cur_destination_station = None
        # -1: not visted, 0: nothing, 1: passenger, 2: destination
        self.station_check = [-1] * 4
        self.in_station = None
        self.passenger_station = None
        self.destination_station = None
        self.last_action = 5

    def update_obs(self, obs):
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

        self.in_station = None

        for i in range(4):
            if taxi_row == stations[i][0] and taxi_col == stations[i][1]:
                self.in_station = i
                break

        if self.in_station is not None:
            self.station_check[self.in_station] = 0
            if passenger_look:
                self.station_check[self.in_station] = 1
                self.passenger_station = self.in_station
            if destination_look:
                self.station_check[self.in_station] = 2
                self.destination_station = self.in_station

    def update_act(self, act):
        self.fuel -= 1
        self.last_action = act
        if self.in_station is not None:
            if (
                act == 4 and self.station_check[self.in_station] == 1
            ):  # If available pick up
                self.have_passenger = True
            elif act == 5 and self.have_passenger:  # If available drop off
                self.have_passenger = False
            if (
                act == 5 and self.station_check[self.in_station] == 2
            ):  # If available drop off -> finish
                self.reset()

        if self.fuel == 0:  # If fuel is 0 -> finish
            self.reset()

    def __call__(self, obs):
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

        passenger_pick_available = (
            not self.have_passenger and self.in_station is not None and passenger_look
        )
        passenger_drop_available = (
            self.have_passenger and self.in_station is not None and destination_look
        )
        self.cur_destination_station = None
        if not self.have_passenger:
            if self.passenger_station is not None:
                self.cur_destination_station = self.passenger_station
        else:
            if self.destination_station is not None:
                self.cur_destination_station = self.destination_station
        if self.cur_destination_station is None:
            self.cur_destination_station = 0
            for i in range(4):
                if self.station_check[i] == -1:
                    self.cur_destination_station = i
                    break

        return (
            stations[self.cur_destination_station][0],
            stations[self.cur_destination_station][1],
            obstacle_north,
            obstacle_south,
            obstacle_east,
            obstacle_west,
            int(passenger_pick_available),
            int(passenger_drop_available),
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


get_key = Get_Key()
