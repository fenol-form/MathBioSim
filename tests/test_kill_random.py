from src import Poisson_1d
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def check(val1, val2, eps=1e-6) -> bool:
    return abs(val1 - val2) <= eps


def test_kill_all():
    death_grid = np.linspace(0.0, 5., num=1001)
    birth_grid = np.linspace(0.5, 1. - 1e-10, num=101)
    sim = Poisson_1d.Poisson_1d(
        area_length_x=np.float_(100.0),
        dd=np.float_(0.01),
        cell_count_x=100,
        b=np.float_(1),
        d=np.float_(0.5),
        initial_population_x=np.arange(10, 90, 1.4),
        seed=1234,
        death_y=stats.norm.pdf(death_grid, scale=1),
        birth_inverse_rcdf_y=stats.norm.ppf(birth_grid, scale=0.2),
        death_cutoff_r=np.float_(5),
        periodic=False,
        realtime_limit=np.float_(60)
    )
    for i in range(250):
        sim.kill_random()
        assert (sim.grid.death_rates >= 0).all()
        assert check(np.sum(sim.grid.death_rates), sim.grid.total_death_rate)

def test_kill_random_cases():
    instances = 10
    death_grid = np.linspace(0.0, 5., num=1001)
    birth_grid = np.linspace(0.5, 1. - 1e-10, num=101)
    for i in range(instances):
        cell_count = stats.randint.rvs(1, 501)
        area_length = stats.uniform.rvs(0, 1001.0)
        left_border = stats.uniform.rvs(0, area_length - 1.)
        right_border = stats.uniform.rvs(left_border + 0.001, area_length - left_border - 0.001)
        individual_amount = stats.randint.rvs(1, 501)
        death_cutoff_r = stats.uniform.rvs(0, area_length / 10)
        sim = Poisson_1d.Poisson_1d(
            area_length_x=area_length,
            dd=np.float_(0.01),
            cell_count_x=cell_count,
            b=np.float_(1),
            d=np.float_(0.5),
            initial_population_x=np.arange(left_border, right_border, (right_border - left_border) / individual_amount),
            seed=1234,
            death_y=stats.norm.pdf(death_grid, scale=1),
            birth_inverse_rcdf_y=stats.norm.ppf(birth_grid, scale=0.2),
            death_cutoff_r=death_cutoff_r,
            periodic=False,
            realtime_limit=np.float_(60)
        )
        for i in range(individual_amount):
            sim.kill_random()
            assert (sim.grid.death_rates >= 0).all()
            assert check(np.sum(sim.grid.death_rates), sim.grid.total_death_rate)