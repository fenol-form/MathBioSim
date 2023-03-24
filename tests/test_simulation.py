from src import Poisson_1d as Poisson_1d
from src import run_simulations
import numpy as np
import scipy.stats as stats


def test_sim():
    death_grid = np.linspace(0.0, 2.95, num=1001)
    birth_grid = np.linspace(0.5, 1. - 1e-10, num=101)
    sim = Poisson_1d.Poisson_1d(
        area_length_x=np.float_(100.0),
        dd=np.float_(0.34),
        cell_count_x=100,
        b=np.float_(0.28),
        d=np.float_(0.08),
        initial_population_x=np.arange(1, 100, 100/18),
        seed=1234,
        death_y=stats.norm.pdf(death_grid, scale=2.95),
        birth_inverse_rcdf_y=stats.norm.ppf(birth_grid, scale=1.63),
        death_cutoff_r=np.float_(2.95),
        periodic=True,
        realtime_limit=np.float_(210)
    )
    result = run_simulations.run_simulations(sim, 600)