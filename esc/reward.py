from scipy.special import expit

from esc.util import MEAN_WATER_TANK_HEIGHT_M

def reward_high_tank_head(tank_head):
    """A function that returns 1 when the tank is full, and 0 when the tank is
    empty, with an S-shaped curve for all values between."""
    return expit(6 * (2 * tank_head / MEAN_WATER_TANK_HEIGHT_M - 1))

def reward_low_energy_cost(energy_cost):
    """A function that returns 1 if energy_cost is 0, and returns reward
    asymptotically approaching 0 as energy cost increases."""
    # return 1 - expit(6 * (2 * (energy_cost + 50) / 400 - 1))
    # return 1.0 / (energy_cost + 1.0) ** 0.25
    return -(1/200) * energy_cost + 1

def reward_few_pump_switches(n_switches):
    return 1 - expit(6 * (2 * n_switches / 50 - 1))