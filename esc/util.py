import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Union, Any

MEAN_WATER_TANK_HEIGHT_M = 3.6576

MINUTES_PER_DAY = 60 * 24

def minute_in_day(t: ArrayLike) -> Union[np.int32, NDArray[np.int32]]:
    """Given a single number, or array of numbers, returns the number(s) rounded
    to the nearest minute of the day they belong to."""
    return np.mod(np.rint(t).astype(np.int32), MINUTES_PER_DAY)
    