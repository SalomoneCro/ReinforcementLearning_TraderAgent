import gymnasium as gym
from spaces import ObservationSpace, ActionSpace
import numpy as np
from datetime import datetime
from utils import get_stocks, get_next_week, prices_observation


class Trader(gym.Env):

    def __init__(self, n_assets, tickers, start_date, end_date, initial_weights, initial_investment):
        self.n_assets = n_assets

        self.observation_space = ObservationSpace(n_assets)
        self.action_space = ActionSpace(n_assets)
        self.observation_space.init_observations()
        self.action_space.init_actions()

        self.tickers = tickers
        
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')

        self.stock_prices = get_stocks(tickers, start_date, end_date)
        self.dates = self.stock_prices.index()
        self.current_date = get_next_week(self.dates, start_date)
        self.shares = (np.array(initial_weights) * initial_investment) / self.stock_prices.loc[start_date]
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
        

    def reset(self):
        self.current_date = get_next_week(self.dates, self.start_date)
        self.liquidity = 0
        self.shares = (np.array(self.initial_weights) * self.initial_investment) / self.stock_prices.loc[self.start_date]


    def step(self, action):
        reward = self.calculate_reward()
        self.rebalance(action)
        
        next_date = get_next_week(self.dates, self.current_date)
        
        obs = self._get_obs(self.current_date, next_date)
        self.current_date = next_date
        
        
        terminated = self.current_date >= self.end_date
        truncated = False
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self, current_date, next_date):
        return prices_observation(current_date, next_date, self.tickers)

    def _get_info(self):
        info = self.simulation.get_info()
        return info

    def calculate_reward(self):
       
        portfolio_value = self.liquidity
        for i in range(len(self.shares)):
            portfolio_value += self.shares[i] * self.stock_prices.loc[self.current_date, self.tickers[i]]

        return portfolio_value


    def rebalance(self, action):
        
        buys = []
        
        # First sell for more liquidity
        for i in range(len(action)):
            if action[i] == -1:
                self.liquidity += self.shares[i] * self.stock_prices.loc[self.current_date, self.tickers[i]]
                self.shares[i] = 0
            elif action[i] == 1:
                buys.append(i)
        
        
        # Then buy
        cash_per_asset = self.liquidity / len(buys)
        for i in buys:
            self.shares[i] += cash_per_asset / self.stock_prices.loc[self.current_date, self.tickers[i]]
        
        if len(buys) != 0:
            self.liquidity = 0

    # def calculate_reward(self):
    #     if self.actual_date != self.end_date:
    #         return 0
    #     else:
    #         reward = self.liquidity_available
            
    #         for i in range(len(self.shares)):
    #             if self.shares[i] > 0:
    #                 reward += self.shares[i] * self.stock_prices.loc[self.current_date, self.tickers[i]]

    #     return reward
