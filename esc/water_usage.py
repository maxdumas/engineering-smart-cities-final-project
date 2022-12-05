"""
Code is adapted from ../notebooks/water-usage.ipynb. See that file for more
explanatory text.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Union, Any
from .util import minute_in_day, MINUTES_PER_DAY


# See ../notebooks/building-heights.ipynb for more information about how these
# numbers were derived.
PER_PERSON_MEAN_WATER_CONSUMPTION_LITER_PER_DAY = 420.18
TYPICAL_BUILDING_N_PEOPLE = 255.68
TYPICAL_BUILDING_MEAN_WATER_CONSUMPTION_LITER_PER_DAY = (
    PER_PERSON_MEAN_WATER_CONSUMPTION_LITER_PER_DAY * TYPICAL_BUILDING_N_PEOPLE
)
TYPICAL_BUILDING_MEAN_WATER_CONSUMPTION_LITER_PER_MINUTE = (
    TYPICAL_BUILDING_MEAN_WATER_CONSUMPTION_LITER_PER_DAY / MINUTES_PER_DAY
)

rng = np.random.default_rng(seed=42)

# A series of approximate characteristic samples of the line in Fig 1b of [this
# paper](https://www.researchgate.net/profile/Rudy-Gargano/publication/268484767_Residential_Water_Demand-Daily_Trends/links/5b4864d50f7e9b4637d22cc2/Residential-Water-Demand-Daily-Trends.pdf).
# The first tuple element is the time of day, second is the relative demand
# level at that time as a multiple of the average demand level.
water_demand_sample = np.array(
    [
        [0, 0.5],
        [3, 0.2],
        [6, 0.25],
        [7, 1.75],
        [12, 1.25],
        [13, 1.5],
        [14, 1.4],
        [16, 0.9],
        [18.5, 1.25],
        [19, 1.75],
        [20, 1.75],
        [24, 0.4],
    ]
)


# Interpolate The sample data to provide a sample per minute and add
# normally-distributed noise to make it more similar to the real world data.
water_demand = np.interp(
    np.linspace(0, 24, MINUTES_PER_DAY),
    water_demand_sample[:, 0],
    water_demand_sample[:, 1],
) + rng.normal(size=MINUTES_PER_DAY, scale=0.05)


def relative_occupant_water_demand(
    t,
):
    """Given a the time of day in minutes t, returns the estimated factor of
    occupant water demand relative to average water demand."""
    return water_demand[minute_in_day(t)]


def typical_building_water_demand(
    t,
):
    """Given a time of day in minutes t, returns the estimated water consumption
    of a typical Manhattan building over 6 stories tall at that time, in liters
    per minute."""
    return (
        TYPICAL_BUILDING_MEAN_WATER_CONSUMPTION_LITER_PER_MINUTE
        * relative_occupant_water_demand(t)
    )
