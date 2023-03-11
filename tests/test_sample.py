from src import Poisson_1d
import numpy as np
import scipy.stats as stats

# content of test_sample.py


def test_sample():
    death_grid = np.linspace(-1, 1, num=100)
    sim = Poisson_1d.Poisson_1d(
        area_length_x=np.float_(100.0),
        dd=np.float_(0.01),
        cell_count_x=100,
        b=np.float_(0.1),
        d=np.float_(0.02),
        initial_population_x=np.linspace(10, 20, 300),
        seed=1234,
        death_y=stats.norm.pdf(death_grid, scale=1),
        birth_inverse_rcdf_y=stats.norm.pdf(death_grid, scale=1),
        death_cutoff_r=np.float_(0.3),
        periodic=False,
        realtime_limit=np.float_(60)
    )
    assert sim.grid.area_length_x == 100.0
    assert len(sim.grid.cell_death_rates) > 0
    assert sim.grid.total_population == 300

