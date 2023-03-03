import numpy as np
import time
from scipy import interpolate

def initialize_death_spline(death_y: np.array, death_cutoff_r: np.float64):
    death_grid = np.linspace(0, death_cutoff_r, len(death_y))
    return interpolate.CubicSpline(death_grid, death_y)


def initialize_ircdf_spline(birth_ircdf_y: np.array):
    birth_ircdf_grid = np.linspace(0, 1.0, len(birth_ircdf_y))
    return interpolate.CubicSpline(birth_ircdf_grid, birth_ircdf_y)


class Cell_1d:
    def __init__(self):
        self.coords_x = np.array([], np.float64)
        self.death_rates = np.array([], np.float64)


class Grid_1d:

    def get_all_x_coords(self, i: int):
        pass

    def get_all_death_rates(self, i: int):
        pass

    def Initialize_death_rates(self):
        # Spawn all speciments
        pass

    def Recalculate_death_rates(self, i: int, spawn: bool):
        pass

    def kill_random(self):
        pass

    def spawn_random(self):
        pass

    def __init__(self,
                 area_length_x: np.float64,
                 cell_count_x: int,
                 b: np.float64,
                 d: np.float64,
                 dd: np.float64,
                 seed: int,
                 initial_population_x: np.array,
                 death_y: np.array,
                 death_cutoff_r: np.float64,
                 birth_inverse_rcdf_y: np.array,
                 periodic: bool,
                 realtime_limit: np.float64
                 ):

        self.area_length_x = area_length_x
        self.cell_count_x = cell_count_x
        self.b = b
        self.d = d
        self.dd = dd
        self.seed = seed

        self.initial_population_x = initial_population_x

        self.death_y = death_y
        self.death_cutoff_r = death_cutoff_r

        self.birth_ircdf_y = birth_inverse_rcdf_y

        self.periodic = periodic
        self.realtime_limit = realtime_limit

        # time
        self.init_time = time.time()

        # splines
        self.death_spline = initialize_death_spline(death_y, death_cutoff_r)
        self.birth_ircdf_spline = initialize_ircdf_spline(birth_inverse_rcdf_y)

        self.death_interaction_cells = max(np.ceil(death_cutoff_r / (area_length_x / cell_count_x)), 3)

        self.cells = np.full(cell_count_x, Cell_1d(), dtype=Cell_1d)
        self.cell_death_rates = np.zeros(cell_count_x, dtype=np.float64)
        self.cell_population = np.zeros(cell_count_x, dtype=int)

        self.Initialize_death_rates()

        # count a total_death_rate
        # ...
