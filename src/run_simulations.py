import numpy as np
import scipy.stats as stats
from src import Poisson_1d


def run_simulations(sim: Poisson_1d.Poisson_1d, epochs: int, calculate_pcf=False, pcf_grid=[]):
    assert epochs > 0
    time = np.zeros(epochs + 1)
    pop = np.zeros(epochs + 1)
    pop[0] = sim.grid.total_population
    time[0] = sim.time

    for i in range(1, epochs + 1):
        sim.run_events(sim.grid.total_population)
        if sim.realtime_limit_reached:
            break
        pop[i] = sim.grid.total_population
        time[i] = sim.time

    return time, pop, sim.realtime_limit_reached
