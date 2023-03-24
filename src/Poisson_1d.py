import numpy as np
import time
from scipy import interpolate
import scipy.stats as stats
from line_profiler_pycharm import profile

def initialize_death_spline(death_y: np.array, death_cutoff_r: np.float64):
    death_grid = np.linspace(0, death_cutoff_r, len(death_y))
    return interpolate.CubicSpline(death_grid, death_y)


def initialize_ircdf_spline(birth_ircdf_y: np.array):
    birth_ircdf_grid = np.linspace(0, 1.0, len(birth_ircdf_y))
    return interpolate.CubicSpline(birth_ircdf_grid, birth_ircdf_y)


def generate_random_index(n: int, weights: np.array) -> int:
    # assert n > 0
    # assert (weights >= 0).all()
    # assert (weights != 0).any()
    pk = weights / np.sum(weights)
    return stats.rv_discrete(values=(np.arange(n), pk)).rvs()


class Grid:

    def __init__(self,
                 cell_count_x: int,
                 periodic: bool,
                 cell_size: int,
                 d: np.float64,
                 area_length_x: np.float64):
        self.cell_count_x = cell_count_x

        # x_coord of each specimen
        self.cell_coords = np.zeros(shape=(cell_count_x, cell_size), dtype=np.float64)

        # death rate of each specimen
        self.cell_death_rates = np.zeros((cell_count_x, cell_size), dtype=np.float64)

        # total death rates in each cell
        self.death_rates = np.zeros(cell_count_x, dtype=np.float64)

        # total population in each cell
        self.cell_population = np.zeros(cell_count_x, dtype=int)

        self.periodic = periodic
        self.total_population = 0
        self.total_death_rate = 0
        self.area_length_x = area_length_x

    def get_storage_shape(self):
        return self.cell_coords.shape

    def get_correct_index(self, i: int) -> int:
        if self.periodic:
            if i < 0:
                i += self.cell_count_x
            if i >= self.cell_count_x:
                i -= self.cell_count_x
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

    @profile
    def add_death_rate(self, target_cell: int, addition: np.array, total_add):
        target_cell = self.get_correct_index(target_cell)
        self.cell_death_rates[target_cell] += addition
        self.death_rates[target_cell] += total_add

        if abs(self.death_rates[target_cell]) < 1e-10:
            self.death_rates[target_cell] = 0

        self.total_death_rate += total_add

    def __extend_grid(self, addition):
        self.cell_coords = np.concatenate([self.cell_coords, np.zeros((self.cell_count_x, addition))], axis=1)
        self.cell_death_rates = np.concatenate([self.cell_death_rates, np.zeros((self.cell_count_x, addition))], axis=1)

    @profile
    def append(self, i: int, x_coord: np.float64, death_rate: np.float64):
        # append specimen at the end of the cell number 'i'
        i = self.get_correct_index(i)

        if self.cell_population[i] >= self.cell_coords.shape[1]:
            self.__extend_grid(self.cell_coords.shape[1])
        j = self.cell_population[i]

        self.cell_coords[i][j] = x_coord
        self.cell_death_rates[i][j] = death_rate

        self.cell_population[i] += 1
        self.total_population += 1

        self.death_rates[i] += death_rate
        self.total_death_rate += death_rate

    @profile
    def remove(self, i: int, j: int, death_rate: np.float64):
        # remove specimen on the position 'j' from the cell number 'i'
        i = self.get_correct_index(i)
        self.cell_population[i] -= 1
        self.total_population -= 1

        self.death_rates[i] -= death_rate
        self.total_death_rate -= death_rate

        if abs(self.death_rates[i]) < 1e-10:
            self.death_rates[i] = 0

        last = self.cell_population[i]
        self.cell_coords[i][j] = self.cell_coords[i][last]
        self.cell_coords[i][last] = 0
        self.cell_death_rates[i][j] = self.cell_death_rates[i][last]
        self.cell_death_rates[i][last] = 0

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
            i = self.grid.get_correct_index(i)
            if (not self.periodic) and (i < 0 or i >= self.cell_count_x):
                continue

            target_spec_inter = np.zeros(self.grid.get_storage_shape()[1])
            vec_f = np.vectorize(self.get_interactions)

            inter = vec_f(self.grid.cell_coords[i], self.grid.cell_coords[target_cell][target_specimen])
            inter[self.grid.cell_population_at(i):] = 0
            if target_cell == i:
                inter[target_specimen] = 0
            total_int = np.sum(inter)
            target_spec_inter[target_specimen] = total_int
            self.grid.add_death_rate(target_cell, target_spec_inter, total_int)

    @profile
    def get_interactions(self, coord, target_coord):
        distance = abs(coord - target_coord)
        if self.periodic:
            distance = min(distance, self.area_length_x - distance)
        if distance > self.death_cutoff_r:
            # too far to interact
            return 0.
        return self.dd * self.death_spline(distance)

    @profile
    def kill_random(self):
        if self.grid.total_population == 0:
            return

        cell_death_index = generate_random_index(self.cell_count_x, self.grid.death_rates)

        in_cell_death_index = generate_random_index(
            self.grid.cell_population_at(cell_death_index),
            self.grid.cell_death_rates[cell_death_index][:self.grid.cell_population_at(cell_death_index)]
        )

        # recalculate death rates
        for i in range(cell_death_index - self.cull_x, cell_death_index + self.cull_x + 1):
            i = self.grid.get_correct_index(i)
            if (not self.periodic) and (i < 0 or i >= self.cell_count_x):
                continue

            removing_spec_inter = np.zeros(shape=self.grid.get_storage_shape()[1])
            vec_f = np.vectorize(self.get_interactions)

            inter = -1 * vec_f(self.grid.cell_coords[i], self.grid.cell_coords[cell_death_index][in_cell_death_index])
            inter[self.grid.cell_population_at(i):] = 0
            if cell_death_index == i:
                inter[in_cell_death_index] = 0
            total_int = np.sum(inter)
            removing_spec_inter[in_cell_death_index] = total_int
            self.grid.add_death_rate(i, inter, total_int)
            self.grid.add_death_rate(cell_death_index, removing_spec_inter, total_int)

        # remove specimen
        self.grid.remove(cell_death_index, in_cell_death_index, self.d)

    @profile
    def spawn_random(self):
        cell_index = generate_random_index(self.cell_count_x, self.grid.cell_population)

        parent_index = generate_random_index(
            self.grid.cell_population_at(cell_index),
            np.ones(self.grid.cell_population_at(cell_index))
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
        index_of_new_spec = self.grid.cell_population_at(new_i) - 1

        # recalculate death rates
        for i in range(new_i - self.cull_x, new_i + self.cull_x + 1):
            i = self.grid.get_correct_index(i)
            if (not self.periodic) and (i < 0 or i >= self.cell_count_x):
                continue

            removing_spec_inter = np.zeros(shape=self.grid.get_storage_shape()[1])
            vec_f = np.vectorize(self.get_interactions)

            inter = vec_f(self.grid.cell_coords[i], self.grid.cell_coords[new_i][index_of_new_spec])
            inter[self.grid.cell_population_at(i):] = 0
            if new_i == i:
                inter[index_of_new_spec] = 0
            total_int = np.sum(inter)
            removing_spec_inter[index_of_new_spec] = total_int
            self.grid.add_death_rate(i, inter, total_int)
            self.grid.add_death_rate(new_i, index_of_new_spec, total_int)

    @profile
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
        self.grid = Grid(cell_count_x, periodic, len(initial_population_x), d, area_length_x)

        self.initialize_death_rates()
