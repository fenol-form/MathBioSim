import numpy as np
import scipy.stats as stats
from src import Poisson_1d


def run_simulations(sim: Poisson_1d.Poisson_1d, epochs: int, calculate_pcf=False, pcf_grid=[]):
    assert epochs > 0
    time = np.zeros(epochs + 1)
    pop = np.zeros(epochs + 1)
    stop = np.zeros(epochs + 1)
    time[0] = sim.time
    pop[0] = sim.grid.total_population
    last_epoch = epochs + 1
    for i in range(1, epochs + 1):
        sim.run_events(sim.grid.total_population)
        time[i] = sim.time
        pop[i] = sim.grid.total_population
        if sim.realtime_limit_reached:
            last_epoch = i
            break
        if sim.grid.total_population == 0:
            last_epoch = i
            break

    if last_epoch < epochs + 1:
        time[last_epoch:epochs + 1] = time[last_epoch]
        pop[last_epoch:epochs + 1] = pop[last_epoch]
        stop[last_epoch:epochs + 1] = 1

    return time, pop, stop
