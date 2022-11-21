"""A module defining electricity rates."""

from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .util import MINUTES_PER_DAY, minute_in_day

EIGHT_AM_MINUTES = 8 * 60

# Taken from https://www.coned.com/en/accounts-billing/your-bill/time-of-use
off_peak_rates = np.ones(EIGHT_AM_MINUTES) * 0.018
peak_rates = np.ones(MINUTES_PER_DAY - EIGHT_AM_MINUTES) * 0.255
all_rates = np.concatenate((off_peak_rates, peak_rates), axis=None)

def electricity_rate(t: ArrayLike) -> Union[np.floating[Any], NDArray[np.floating[Any]]]:
    return all_rates[minute_in_day(t)]