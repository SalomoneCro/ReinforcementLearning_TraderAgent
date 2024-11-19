from agent import Trader
from utils import get_stocks


tickers = ['AAPL', 'NVDA', 'ORCL']
start_date = '2024-01-01'
end_date = '2024-03-01'
initial_weights = [0.0,0.0,1.0]
initial_investment = 100
n_assets = len(tickers)
stock_prices = get_stocks(tickers, start_date, end_date)

bot = Trader(stock_prices, n_assets, tickers, initial_weights, initial_investment)


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