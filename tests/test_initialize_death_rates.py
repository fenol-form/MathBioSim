from src import Poisson_1d
import numpy as np
import scipy.stats as stats
import pandas as pd

death_grid = np.linspace(0.0, 0.02 * 3., num=1000)
birth_grid = np.linspace(0.5, 1. - 1e-10, num=1000)


def init_simulator(init_pop):
    sim = Poisson_1d.Poisson_1d(
        area_length_x=np.float_(100.0),
        dd=np.float_(0.001),
        cell_count_x=100,
        b=np.float_(0.4),
        d=np.float_(0.2),
        initial_population_x=init_pop,
        seed=1234,
        death_y=stats.norm.pdf(death_grid, scale=0.02),
        birth_inverse_rcdf_y=stats.norm.ppf(birth_grid, scale=0.12),
        death_cutoff_r=np.float_(0.02 * 3),  # death_y scale * 3
        periodic=True,
        realtime_limit=np.float_(60)
    )
    return sim


def test_1():
    sim = init_simulator(np.arange(start=0, stop=100.0, step=5.))

    death_y = pd.DataFrame()
    death_y['actual'] = pd.read_csv("data/death_y_sd002-1.csv")
    death_y['new'] = sim.death_y
    death_y['diff'] = abs(death_y['actual'] - death_y['new'])
    assert (death_y['diff'] <= 0.03).all()

    assert len(sim.initial_population_x) == 20

    x_coordinate = pd.DataFrame()
    x_coordinate['actual'] = pd.read_csv('data/all_x_coord-1.csv')
    x_coordinate['new'] = sim.grid.get_all_coords()

    cell_death_rates = pd.DataFrame()
    cell_death_rates['actual'] = pd.read_csv('data/all_death_rates-1.csv')
    cell_death_rates['new'] = sim.grid.get_all_death_rates()
    print(cell_death_rates.head(20))


def test_2():
    sim = init_simulator(np.arange(start=0, stop=100.0, step=2.))

    death_y = pd.DataFrame()
    death_y['actual'] = pd.read_csv("data/death_y_sd002-2.csv")
    death_y['new'] = sim.death_y
    death_y['diff'] = abs(death_y['actual'] - death_y['new'])
    assert (death_y['diff'] <= 0.03).all()

    assert len(sim.initial_population_x) == 50

    x_coordinate = pd.DataFrame()
    x_coordinate['actual'] = pd.read_csv('data/all_x_coord-2.csv')
    x_coordinate['new'] = sim.grid.get_all_coords()

    cell_death_rates = pd.DataFrame()
    cell_death_rates['actual'] = pd.read_csv('data/all_death_rates-2.csv')
    cell_death_rates['new'] = sim.grid.get_all_death_rates()
    print(cell_death_rates.head(20))


def test_3():
    init_pop = np.hstack([
        np.arange(2, 5, 0.7),
        np.arange(40, 45, 0.2),
        np.arange(80, 100, 1.4)
    ])
    sim = init_simulator(init_pop)

    death_y = pd.DataFrame()
    death_y['actual'] = pd.read_csv("data/death_y_sd002-3.csv")
    death_y['new'] = sim.death_y
    death_y['diff'] = abs(death_y['actual'] - death_y['new'])
    assert (death_y['diff'] <= 0.03).all()

    assert len(sim.initial_population_x) == len(init_pop)

    x_coordinate = pd.DataFrame()
    x_coordinate['actual'] = pd.read_csv('data/all_x_coord-3.csv')
    x_coordinate['new'] = sim.grid.get_all_coords()
    print(x_coordinate.head(20))

    cell_death_rates = pd.DataFrame()
    cell_death_rates['actual'] = pd.read_csv('data/all_death_rates-3.csv')
    cell_death_rates['new'] = sim.grid.get_all_death_rates()


