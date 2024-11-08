from GettingData import get_adj_close_on_date

start_date = "2024-05-01"
end_date = "2024-05-07"

tickers = ['AAPL', 'MSFT', 'GOOGL']

def tickers_dinamycs(start_date, end_date, tickers):
    prices0 = get_adj_close_on_date(tickers=tickers, date=start_date)
    prices1 = get_adj_close_on_date(tickers=tickers, date=end_date)

    comparison_dict = {key: 1 if prices1[key] > prices0[key] else -1 for key in tickers}

    return comparison_dict
