import gymnasium as gym
from spaces import ObservationSpace, ActionSpace
import numpy as np
from datetime import datetime
from utils import get_next_week, prices_observation


class Trader(gym.Env):

    def __init__(
            self, 
            stock_prices, 
            n_assets, 
            tickers, 
            initial_weights, 
            initial_investment
        ):

        self.observation_space = ObservationSpace(n_assets)
        self.action_space = ActionSpace(n_assets)

        self.tickers = tickers
        self.n_assets = len(tickers)
        

        self.stock_prices = stock_prices
        self.dates = self.stock_prices.index
        self.start_date = self.dates[0]
        self.end_date = self.dates[-1]
        
        self.current_date = get_next_week(self.dates, self.start_date)
        self.shares = (np.array(initial_weights) * initial_investment) / self.stock_prices.loc[self.start_date]
        self.liquidity = 0
        self.initial_weights = initial_weights
        self.initial_investment = initial_investment

        """
        Variables for rendering
        """
        self.render_mode = None
        self.window = None
        self.clock = None
        self._elapsed_steps = None      
        

    def reset(self, **kwargs):
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)  # Establecer semilla para reproducibilidad

        # Reiniciar el estado del entorno
        self.current_date = get_next_week(self.dates, self.start_date)
        self.liquidity = 0
        self.shares = (np.array(self.initial_weights) * self.initial_investment) / self.stock_prices.loc[self.start_date]

        # Devolver la observación inicial como un arreglo NumPy
        obs = self._get_obs(self.start_date, self.current_date)
        
        return obs, {}


    def step(self, action):
        # Adaption because of the MultiDiscrete type
        action = action - np.ones(len(action)) 

        self.rebalance(action)

        next_date = get_next_week(self.dates, self.current_date)

        obs = self._get_obs(self.current_date, next_date)
        self.current_date = next_date
        reward = self.calculate_reward()

        current_date = datetime.strptime(self.current_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
        terminated = current_date >= end_date
        truncated = False
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self, current_date, next_date):
        obs = prices_observation(current_date, next_date, self.stock_prices)
        obs = np.array(list(obs.values()), dtype=np.float32)
        return obs

    def _get_info(self):
        return dict(zip(self.tickers, self.shares))

    def calculate_reward(self):
       
        portfolio_value = self.liquidity
        for i in range(len(self.shares)):
            portfolio_value += self.shares[self.tickers[i]] * self.stock_prices.loc[self.current_date, self.tickers[i]]

        return portfolio_value


    def rebalance(self, action):
        buys = []

        # First sell for more liquidity (validate actions)
        for i in range(len(action)):
            if action[i] == -1:
                if self.shares[self.tickers[i]] > 0:  # Only sell if there are shares
                    self.liquidity += self.shares[self.tickers[i]] * self.stock_prices.loc[self.current_date, self.tickers[i]]
                    self.shares[self.tickers[i]] = 0
            elif action[i] == 1:
                buys.append(i)

        # Then buy (validate actions)
        if len(buys) != 0:
            cash_per_asset = self.liquidity / len(buys)

            for i in buys:
                if cash_per_asset > 0:  # Only buy if there is liquidity
                    self.shares[self.tickers[i]] += cash_per_asset / self.stock_prices.loc[self.current_date, self.tickers[i]]

            self.liquidity = 0