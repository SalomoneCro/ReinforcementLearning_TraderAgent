from gymnasium.spaces import Box
import numpy as np
from gymnasium.spaces import MultiDiscrete

class ObservationSpace(Box):
    """
    The observations are the information if the asset went down
    or up. An observation is an n-array where n is the number
    of assets in the portfolio. The array contains only 1 and -1.
    If the asset i went up, the i-th element is 1, otherwise is
    -1. 
    """

    def __init__(self, n_assets):
        low = np.full(n_assets, -1, dtype=np.int8)
        high = np.full(n_assets, 1, dtype=np.int8)
        super().__init__(low, high, dtype=np.int8)


class ActionSpace(MultiDiscrete):
    """
    The actions are sell, hold and buy (-1, 0, 1 respectively).
    Each actions is an n-array containing the order for each asset
    where n is the number of assets in the portfolio
    """

    def __init__(self, n_assets):
        super().__init__([3] * n_assets)