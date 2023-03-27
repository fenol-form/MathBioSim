from src import Poisson_1d
from src import run_simulations
import numpy as np
import scipy.stats as stats
from line_profiler_pycharm import profile


@profile
def test_sim():
    death_grid = np.linspace(0.0, 3.198, num=1001)
    birth_grid = np.linspace(0.5, 1. - 1e-10, num=101)
    sim = Poisson_1d.Poisson_1d(
        area_length_x=100,
        dd = 0.4121996,
        d = 0.135,
        b = 0.57,
        dsd = 3.198,
        bsd = 0.347,
        initial_population_x = np.arange(1, 100, 100 / 113),
        death_cutoff_r = 3.198,
        seed=42,
        periodic=True,
        realtime_limit=30
    )
    result = run_simulations.run_simulations(sim, 600)



