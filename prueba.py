from spaces import ObservationSpace, ActionSpace
from agent import Trader



tickers = ['AAPL', 'NVDA', 'ORCL']
start_date = '2024-01-01'
end_date = '2024-03-01'
initial_weights = [0.0,0.0,1.0]
initial_investment = 100
n_assets = len(tickers)

bot = Trader(n_assets, tickers, start_date, end_date, initial_weights, initial_investment)


print(len(bot.dates))
print(bot.stock_prices.loc['2024-01-02'])
print(bot.shares)
print(bot.stock_prices.loc['2024-01-09'])
obs, reward, terminated, truncated, info = bot.step((0,0,-1))
print(obs, reward, terminated, truncated, info)
print(bot.shares)
obs, reward, terminated, truncated, info = bot.step((0,0,-1))
print(obs, reward, terminated, truncated, info)
print(bot.shares)