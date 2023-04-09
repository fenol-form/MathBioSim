from src import Poisson_1d


def run_simulations(sim: Poisson_1d.Poisson_1d, epochs: int, calculate_pcf=False, pcf_grid=[]):
    assert epochs > 0
    time = [sim.grid.total_population]
    pop = [sim.time]

    for i in range(1, epochs + 1):
        sim.run_events(sim.grid.total_population)
        if sim.realtime_limit_reached:
            break
        pop.append(sim.grid.total_population)
        time.append(sim.time)

    return time, pop, sim.realtime_limit_reached
