import numpy as np
import time
from scipy import interpolate
from copy import deepcopy, copy


def initialize_death_spline(death_y: np.array, death_cutoff_r: np.float64):
    death_grid = np.linspace(0, death_cutoff_r, len(death_y))
    return interpolate.CubicSpline(death_grid, death_y)


def initialize_ircdf_spline(birth_ircdf_y: np.array):
    birth_ircdf_grid = np.linspace(0, 1.0, len(birth_ircdf_y))
    return interpolate.CubicSpline(birth_ircdf_grid, birth_ircdf_y)


class Grid:

    def __init__(self,
                 cell_count_x: int,
                 periodic: bool,
                 cell_size: int,
                 d: np.float64,
                 area_length_x: np.float64):
        self.cell_count_x = cell_count_x

        # x_coord of each specimen
        self.cell_coords = []

        # death rate of each specimen
        self.cell_death_rates = []

        # total death rates in each cell
        self.death_rates = np.zeros(cell_count_x, dtype=np.float64)

        # total population in each cell
        self.cell_population = np.zeros(cell_count_x, dtype=int)

        for i in range(cell_count_x):
            self.cell_coords.append(np.zeros(cell_size, dtype=np.float64))
            self.cell_death_rates.append(np.full(cell_size, d, dtype=np.float64))

        self.periodic = periodic
        self.total_population = 0
        self.total_death_rate = 0
        self.area_length_x = area_length_x

    def get_correct_index(self, i: int) -> int:
        if self.periodic:
            if i < 0:
                i += self.cell_count_x
            if i >= self.cell_count_x:
                i -= self.cell_count_x
        assert 0 <= i < self.cell_count_x
        return i

    def cell_population_at(self, i: int) -> int:
        i = self.get_correct_index(i)
        return self.cell_population[i]

    def cell_death_rate_at(self, i: int) -> np.float64:
        i = self.get_correct_index(i)
        return self.death_rates[i]

    def get_death_rates(self, i: int) -> np.array:
        i = self.get_correct_index(i)
        return self.cell_death_rates[i]

    def get_cell_coords(self, i: int) -> np.array:
        i = self.get_correct_index(i)
        return self.cell_coords[i]

    def add_death_rate(self, target_cell: int, target_specimen: int, death_rate: np.float64):

        # add death_rate to the target_specimen being in target_cell
        target_cell = self.get_correct_index(target_cell)
        self.cell_death_rates[target_cell][target_specimen] += death_rate
        self.death_rates[target_cell] += death_rate
        self.total_death_rate += death_rate

    def append(self, i: int, x_coord: np.float64, death_rate: np.float64):

        # append specimen at the end of the cell number 'i'
        i = self.get_correct_index(i)
        self.cell_coords[i] = np.append(self.cell_coords[i], x_coord)
        self.cell_population[i] += 1
        self.cell_death_rates[i] = np.append(self.cell_death_rates[i], death_rate)
        self.death_rates[i] += death_rate
        self.total_population += 1
        self.total_death_rate += death_rate

    def remove(self, i: int, j: int, death_rate: np.float64):
        # remove specimen on the position 'j' from the cell number 'i'
        pass

    def count_distance(self, cell_i: int, cell_j: int, i: int, j: int) -> np.float64:
        distance = 0.0
        cell_i = self.get_correct_index(cell_i)
        cell_j = self.get_correct_index(cell_j)
        if self.periodic:
            # TO DO
            return 0.0
        else:
            return abs(self.cell_coords[cell_i][i] - self.cell_coords[cell_j][j])


class Poisson_1d:

    def initialize_death_rates(self):

        # Spawn all specimens
        for x_coord in self.initial_population_x:
            if x_coord < 0 or x_coord > self.area_length_x:
                continue
            i = int(np.floor(x_coord * self.cell_count_x / self.area_length_x))

            if i >= self.cell_count_x:
                i -= 1

            self.grid.append(i, x_coord, self.d)

        # Recalculate death rates
        for i in range(self.cell_count_x):
            for k in range(self.grid.cell_population_at(i)):
                self.recalculate_death_rates(i, k)

    def recalculate_death_rates(self, target_cell: int, target_specimen: int):
        for i in range(target_cell - self.cull_x, target_cell + self.cull_x + 1):
            if (not self.periodic) and (i < 0 or i >= self.cell_count_x):
                continue

            for p in range(self.grid.cell_population_at(i)):
                if target_cell == i and p == target_specimen:
                    continue

                distance = self.grid.count_distance(i, target_cell, p, target_specimen)

                if distance > self.death_cutoff_r:
                    # too far to interact
                    continue

                interaction = self.dd * self.death_spline(distance)
                self.grid.add_death_rate(target_cell, target_specimen, interaction)

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

        self.init_time = time.time()

        self.death_spline = initialize_death_spline(death_y, death_cutoff_r)
        self.birth_ircdf_spline = initialize_ircdf_spline(birth_inverse_rcdf_y)

        self.cull_x = max(np.ceil(death_cutoff_r / (area_length_x / cell_count_x)), 3)
        self.grid = Grid(cell_count_x, periodic, 0, d, area_length_x)

        self.initialize_death_rates()

        # count a total_death_rate
        # ...
