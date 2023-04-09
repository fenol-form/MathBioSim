import numpy as np
import time
from scipy import interpolate
from copy import deepcopy, copy
import scipy.stats as stats

def initialize_death_spline(death_y: np.array, death_cutoff_r: np.float64):
    death_grid = np.linspace(0, death_cutoff_r, len(death_y))
    return interpolate.CubicSpline(death_grid, death_y)


def initialize_ircdf_spline(birth_ircdf_y: np.array):
    birth_ircdf_grid = np.linspace(0, 1.0, len(birth_ircdf_y))
    return interpolate.CubicSpline(birth_ircdf_grid, birth_ircdf_y)


def generate_random_index(n: int, weights: np.array) -> int:
    assert n > 0
    assert (weights >= 0).all()
    assert (weights != 0).any()

    pk = weights / np.sum(weights)
    try:
        index = stats.rv_discrete(values=(np.arange(n), pk)).rvs()
    except ValueError:
        pk[-1] = 1. - np.sum(pk[:-1])
        index = stats.rv_discrete(values=(np.arange(n), pk)).rvs()

    assert 0 <= index < n
    return index


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

    def get_all_coords(self) -> np.array:
        x_coords = np.array([])
        for lst in self.cell_coords:
            if len(lst) > 0:
                x_coords = np.hstack([x_coords, lst])
        return x_coords

    def get_all_death_rates(self) -> np.array:
        all_death_rates = np.array([])
        for lst in self.cell_coords:
            if len(lst) > 0:
                all_death_rates = np.hstack([all_death_rates, lst])
        return all_death_rates

    def add_death_rate(self, target_cell: int, target_specimen: int, death_rate: np.float64):
        # add death_rate to the target_specimen being in target_cell
        target_cell = self.get_correct_index(target_cell)

        self.cell_death_rates[target_cell][target_specimen] += death_rate
        self.death_rates[target_cell] += death_rate

        if abs(self.death_rates[target_cell]) < 1e-10:
            self.death_rates[target_cell] = 0

        assert self.death_rates[target_cell] >= 0

        self.total_death_rate += death_rate

    def append(self, i: int, x_coord: np.float64, death_rate: np.float64):
        # append specimen at the end of the cell number 'i'
        i = self.get_correct_index(i)
        self.cell_population[i] += 1
        self.total_population += 1

        self.death_rates[i] += death_rate
        self.total_death_rate += death_rate

        self.cell_coords[i] = np.append(self.cell_coords[i], x_coord)
        self.cell_death_rates[i] = np.append(self.cell_death_rates[i], death_rate)

    def remove(self, i: int, j: int, death_rate: np.float64):
        # remove specimen on the position 'j' from the cell number 'i'
        i = self.get_correct_index(i)
        assert 0 <= j < len(self.cell_death_rates[i])
        self.cell_population[i] -= 1
        self.total_population -= 1

        self.death_rates[i] -= death_rate
        self.total_death_rate -= death_rate

        if abs(self.death_rates[i]) < 1e-10:
            self.death_rates[i] = 0

        self.cell_coords[i] = np.delete(self.cell_coords[i], j)
        self.cell_death_rates[i] = np.delete(self.cell_death_rates[i], j)

    def count_distance(self, cell_i: int, cell_j: int, i: int, j: int) -> np.float64:
        cell_i = self.get_correct_index(cell_i)
        cell_j = self.get_correct_index(cell_j)
        distance = abs(self.cell_coords[cell_i][i] - self.cell_coords[cell_j][j])
        if self.periodic:
            return min(distance, self.area_length_x - distance)
        else:
            return distance


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

    def kill_random(self, target_cell=-1, target_specimen=-1):
        if self.grid.total_population == 0:
            return

        # generate dying specimen
        cell_death_index = generate_random_index(self.cell_count_x, self.grid.death_rates)

        assert self.grid.cell_population[cell_death_index] > 0

        if self.grid.cell_population[cell_death_index] <= 0:
                return
        in_cell_death_index = generate_random_index(
            len(self.grid.cell_death_rates[cell_death_index]),
            self.grid.cell_death_rates[cell_death_index]
        )

        # recalculate death rates
        for i in range(cell_death_index - self.cull_x, cell_death_index + self.cull_x + 1):
            if (not self.periodic) and (i < 0 or i >= self.cell_count_x):
                continue

            for p in range(self.grid.cell_population_at(i)):
                if cell_death_index == i and p == in_cell_death_index:
                    continue

                distance = self.grid.count_distance(i, cell_death_index, p, in_cell_death_index)

                if distance > self.death_cutoff_r:
                    # too far to interact
                    continue

                interaction = self.dd * self.death_spline(distance)
                self.grid.add_death_rate(i, p, -1 * interaction)
                self.grid.add_death_rate(cell_death_index, in_cell_death_index, -1 * interaction)

        # remove specimen
        self.grid.remove(cell_death_index, in_cell_death_index, self.d)

    def spawn_random(self):
        # generate parent specimen
        cell_index = generate_random_index(self.cell_count_x, self.grid.cell_population)
        parent_index = generate_random_index(
            len(self.grid.cell_coords[cell_index]),
            np.ones(len(self.grid.cell_coords[cell_index]))
        )
        new_coord_x = self.grid.cell_coords[cell_index][parent_index] + \
            self.birth_ircdf_spline(stats.uniform.rvs()) * (2. * stats.bernoulli.rvs(0.5) - 1.)

        if new_coord_x < 0 or new_coord_x > self.area_length_x:
            if not self.periodic:
                # Specimen failed to spawn and died outside area boundaries
                return
            if new_coord_x < 0:
                new_coord_x += self.area_length_x
            if new_coord_x > self.area_length_x:
                new_coord_x -= self.area_length_x

        new_i = int(np.floor(new_coord_x * self.cell_count_x / self.area_length_x))
        if new_i == self.cell_count_x:
            new_i -= 1

        # New specimen is added to the end of array
        self.grid.append(new_i, new_coord_x, self.d)
        index_of_new_spec = len(self.grid.cell_coords[new_i]) - 1

        # recalculate death rates
        for i in range(new_i - self.cull_x, new_i + self.cull_x + 1):
            if (not self.periodic) and (i < 0 or i >= self.cell_count_x):
                continue

            for p in range(self.grid.cell_population_at(i)):
                if new_i == i and p == index_of_new_spec:
                    continue

                distance = self.grid.count_distance(i, new_i, p, index_of_new_spec)

                if distance > self.death_cutoff_r:
                    # too far to interact
                    continue

                interaction = self.dd * self.death_spline(distance)
                self.grid.add_death_rate(i, p, interaction)
                self.grid.add_death_rate(new_i, index_of_new_spec, interaction)

    def make_event(self):
        if self.grid.total_population == 0:
            return -1
        self.event_count += 1
        self.time += stats.expon.rvs(scale=1. / (self.grid.total_population * self.b + self.grid.total_death_rate))
        born_probability = self.grid.total_population * self.b / \
                            (self.grid.total_population * self.b + self.grid.total_death_rate)
        if stats.bernoulli.rvs(born_probability) == 0:
            self.kill_random()
        else:
            self.spawn_random()

    def run_events(self, events: int):
        if events > 0:
            for i in range(events):
                if time.time() > self.init_time + self.realtime_limit:
                    self.realtime_limit_reached = True
                    return
                self.make_event()

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
        np.random.seed(seed)

        self.initial_population_x = initial_population_x

        self.death_y = death_y
        self.death_cutoff_r = death_cutoff_r

        self.birth_ircdf_y = birth_inverse_rcdf_y

        self.periodic = periodic
        self.realtime_limit = realtime_limit

        self.init_time = time.time()
        self.time = 0.
        self.realtime_limit_reached = False
        self.event_count = 0

        self.death_spline = initialize_death_spline(death_y, death_cutoff_r)
        self.birth_ircdf_spline = initialize_ircdf_spline(birth_inverse_rcdf_y)

        self.cull_x = max(int(death_cutoff_r / (area_length_x / cell_count_x)), 3)
        self.grid = Grid(cell_count_x, periodic, 0, d, area_length_x)

        self.initialize_death_rates()
