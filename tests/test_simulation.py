from src import Poisson_1d
from src import run_simulations
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def test_spawn_random():
    death_grid = np.linspace(0.0, 5., num=1001)
    birth_grid = np.linspace(0.5, 1. - 1e-10, num=101)
    sim = Poisson_1d.Poisson_1d(
        area_length_x=np.float_(100.0),
        dd=np.float_(0.01),
        cell_count_x=100,
        b=np.float_(1),
        d=np.float_(0.5),
        initial_population_x=np.arange(0, 50, 1),
        seed=1234,
        death_y=stats.norm.pdf(death_grid, scale=1),
        birth_inverse_rcdf_y=stats.norm.ppf(birth_grid, scale=0.2),
        death_cutoff_r=np.float_(5),
        periodic=True,
        realtime_limit=np.float_(60)
    )
    td = sim.grid.total_death_rate
    sim.spawn_random()
    sim.spawn_random()
    assert sim.grid.total_death_rate >= td
    assert sim.grid.total_population == 52

    for i in range(100):
        td = sim.grid.total_death_rate
        sim.spawn_random()
        assert sim.grid.total_death_rate >= td


def test_sim():
    death_grid = np.linspace(0.0, 5., num=1001)
    birth_grid = np.linspace(0.5, 1. - 1e-10, num=101)
    sim = Poisson_1d.Poisson_1d(
        area_length_x=np.float_(100.0),
        dd=np.float_(0.01),
        cell_count_x=100,
        b=np.float_(1),
        d=np.float_(0.),
        initial_population_x=[10.],
        seed=1234,
        death_y=stats.norm.pdf(death_grid, scale=1),
        birth_inverse_rcdf_y=stats.norm.ppf(birth_grid, scale=0.2),
        death_cutoff_r=np.float_(5),
        periodic=True,
        realtime_limit=np.float_(60)
    )
    result = run_simulations.run_simulations(sim, 100)
