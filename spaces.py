from gymnasium.spaces import Space
import itertools

class ObservationSpace(Space):
    """
    The observations are the information if the asset went down
    or up. An observation is an n-array where n is the number
    of assets in the portfolio. The array contains only 1 and -1.
    If the asset i went up, the i-th element is 1, otherwise is
    -1. 
    """

    def __init__(self, n_assets) -> None:
        self.n_assets = n_assets

    def init_observations(self) -> None:
        """
        Generates all possible n-tuples of -1 and 1.
        """
        self.observations = list(itertools.product([1, -1], repeat=self.n_assets))


class ActionSpace(Space):
    """
    The actions are sell, hold and buy (-1, 0, 1 respectively).
    Each actions is an n-array containing the order for each asset
    where n is the number of assets in the portfolio
    """

    def __init__(self, n_assets) -> None:
        self.n_assets = n_assets

    def init_actions(self) -> None:
        """
        Generates all possible n-tuples of -1, 0, 1.
        """
        self.actions = list(itertools.product([1, -1, 0], repeat=self.n_assets))