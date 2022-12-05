"""A module defining electricity rates."""

from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .util import minute_in_day

# See electricity_rates.ipynb for how this was generated
all_rates = np.load("../data/nyc_price_by_minute_of_day_2022-08-01.npy")

def electricity_rate(t):
    return all_rates[minute_in_day(t)]