import numpy as np
import time
import scipy.stats as stats

import jax.numpy as jnp
from jax import vmap, jit

from jax.config import config
from functools import partial

config.update("jax_enable_x64", True)


@partial(jit, static_argnames=['area_length_x', 'periodic'])
def calculate_distances(grid_slice: jnp.array, target_coord, area_length_x, periodic):
    def _get_distance(coord, target_coord):
        return abs(coord - target_coord)

    def _get_distance_bounds(coord, target_coord, area_length_x):
        return abs(area_length_x - abs(coord - target_coord))

    vf = vmap(_get_distance, (0, None))
    dist = vf(grid_slice, target_coord)
    if periodic:
        vf_b = vmap(_get_distance_bounds, (0, None, None))
        dist_bounds = vf_b(grid_slice, target_coord, area_length_x)
        dist = jnp.minimum(dist, dist_bounds)

    return dist


@partial(jit, static_argnames=['dsd', 'death_r', 'dd'])
def calculate_interaction(distances: jnp.array, dsd, death_r, dd):
    diff = death_r - distances
    # condition = 0 when it's too far to interact
    # condition = 1 otherwise
    condition = jnp.nan_to_num(jnp.multiply(distances, jnp.maximum(diff, 0)) / diff, nan=death_r)
    condition = jnp.nan_to_num(condition / condition, nan=0)

    def _get_inter(dist, cond):
        return (1. / jnp.sqrt(2. * jnp.pi) / dsd) * jnp.exp(-1. * (dist * dist) / 2. / dsd / dsd) * dd * cond

    vf = vmap(_get_inter, (0, 0))
    return _get_inter(distances, condition)


def generate_random_index(weights: jnp.array) -> int:
    return np.random.choice(np.arange(weights.shape[0]), p=weights / np.sum(weights))


class Grid:

    def __init__(self,
                 periodic: bool,
                 init_size: int,
                 area_length_x: np.float64):
        self.unexisted_x = 10 * area_length_x

        # x_coord of each specimen
        self.x_coords = np.full(shape=(init_size), fill_value=self.unexisted_x, dtype=np.float64)

        # death rate of each specimen
        self.death_rates = np.zeros(init_size, dtype=np.float64)

        self.periodic = periodic
        self.total_population = 0
        self.total_death_rate = 0
        self.area_length_x = area_length_x

    def get_storage_shape(self):
        return self.x_coords.shape

    def add_death_rate_point(self, index: int, addition: np.float_):
        self.death_rates[index] += addition
        self.total_death_rate += addition

    def add_death_rates(self, addition: np.array, total_add=None):
        self.death_rates += np.array(addition)
        if total_add is None:
            total_add = jnp.sum(addition)
        self.total_death_rate += total_add

    def __extend_grid(self, addition):
        self.x_coords = np.concatenate([self.x_coords,
                                        np.full(addition, fill_value=self.unexisted_x)])
        self.death_rates = np.concatenate([self.death_rates, np.zeros(addition)])

    def append(self, x_coord: np.float64, death_rate: np.float64):
        # append specimen at the end of the grid
        if self.total_population >= self.x_coords.shape[0]:
            self.__extend_grid(self.x_coords.shape[0])
        j = self.total_population

        self.x_coords[j] = x_coord
        self.death_rates[j] = death_rate

        self.total_population += 1

        self.total_death_rate += death_rate

    def remove(self, j: int, death_rate: np.float64):
        # remove specimen on the position 'j'
        self.total_population -= 1

        self.total_death_rate -= death_rate

        last = self.total_population
        self.x_coords[j] = self.x_coords[last]
        self.x_coords[last] = self.unexisted_x
        self.death_rates[j] = self.death_rates[last]
        self.death_rates[last] = 0

    def count_distance(self, i: int, j: int) -> np.float64:
        distance = abs(self.x_coords[i] - self.x_coords[j])
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

            self.grid.append(x_coord, self.d)

        # Recalculate death rates
        for target_specimen in range(self.grid.total_population):
            for p in range(self.grid.total_population):
                if p == target_specimen:
                    continue
                distance = self.grid.count_distance(p, target_specimen)
                if distance > self.death_cutoff_r:
                    # too far to interact
                    continue
                interaction = self.dd * stats.norm.pdf(distance, scale=self.dsd)
                self.grid.add_death_rate_point(target_specimen, interaction)

    def kill_random(self):
        if self.grid.total_population == 0:
            return

        death_index = generate_random_index(
            self.grid.death_rates
        )

        # recalculate death rates
        target_coord = self.grid.x_coords[death_index]
        distances = calculate_distances(self.grid.x_coords, target_coord, self.area_length_x, self.periodic)
        inter = -1 * calculate_interaction(distances, self.dsd, self.death_cutoff_r, self.dd)

        total_add = jnp.sum(inter)
        self.grid.add_death_rate_point(death_index, total_add)
        self.grid.add_death_rates(inter, total_add)

        # remove specimen
        self.grid.remove(death_index, self.d)

    def spawn_random(self):
        parent_index = generate_random_index(
            np.ones(self.grid.total_population)
        )

        shift = np.random.normal(scale=self.dsd) * (2. * np.random.randint(low=0, high=1) - 1.)
        new_coord_x = self.grid.x_coords[parent_index] + shift

        if new_coord_x < 0 or new_coord_x > self.area_length_x:
            if not self.periodic:
                # specimen failed to spawn and died outside area boundaries
                return
            if new_coord_x < 0:
                new_coord_x += self.area_length_x
            if new_coord_x > self.area_length_x:
                new_coord_x -= self.area_length_x

        # new specimen is added to the end of array
        self.grid.append(new_coord_x, self.d)
        index_of_new_spec = self.grid.total_population - 1

        # recalculate death rates
        distances = calculate_distances(self.grid.x_coords, new_coord_x, self.area_length_x, self.periodic)
        inter = calculate_interaction(distances, self.dsd, self.death_cutoff_r, self.dd)

        total_add = jnp.sum(inter)
        self.grid.add_death_rate_point(index_of_new_spec, total_add)
        self.grid.add_death_rates(inter, total_add)

    def make_event(self):
        if self.grid.total_population == 0:
            return -1
        self.event_count += 1
        self.time += np.random.exponential(scale=1. / (self.grid.total_population * self.b + self.grid.total_death_rate))
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
                 b: np.float64,
                 d: np.float64,
                 dd: np.float64,
                 seed: int,
                 initial_population_x: np.array,
                 dsd: np.float64,
                 bsd: np.float64,
                 death_cutoff_r: np.float64,
                 periodic: bool,
                 realtime_limit: np.float64
                 ):

        self.area_length_x = area_length_x
        self.b = b
        self.d = d
        self.dd = dd
        self.seed = seed
        self.dsd = dsd
        self.bsd = bsd
        np.random.seed(seed)

        self.initial_population_x = initial_population_x

        # require death_cutoff_r < area_length_x
        self.death_cutoff_r = death_cutoff_r

        self.periodic = periodic
        self.realtime_limit = realtime_limit

        self.init_time = time.time()
        self.time = 0.
        self.realtime_limit_reached = False
        self.event_count = 0

        self.grid = Grid(periodic, len(initial_population_x), area_length_x)

        self.initialize_death_rates()
