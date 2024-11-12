from spaces import ObservationSpace, ActionSpace
from agent import Trader

obs = ObservationSpace(3)
actions = ActionSpace(3)

print(actions.sample())

tickers = ['AAPL', 'NVDA', 'ORCL']
start_date = '2024-01-01'
end_date = '2024-03-01'
initial_weights = [0.5, 0.3, 0.2]
initial_investment = 100
n_assets = len(tickers)

bot = Trader(n_assets, tickers, start_date, end_date, initial_weights, initial_investment)

print(bot.tickers)
print(bot.n_assets)
print(bot.start_date)
print(bot.end_date)
print(bot.dates)
print(bot.current_date)
print(bot.shares)
print(bot.liquidity)
print(bot.initial_weights)
print(bot.initial_investment)