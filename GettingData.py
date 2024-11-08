import yfinance as yf
import datetime
import warnings

def get_next_open_market_date(date: str) -> str:
    """
    Given a date, returns the date if the market was open on that day.
    Otherwise, returns the next open market day.

    Parameters:
    - date (str): The date to check in 'YYYY-MM-DD' format.

    Returns:
    - str: The next open market date in 'YYYY-MM-DD' format.
    """
    date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
    max_attempts = 10  # Limit to avoid infinite loops in case of an error

    for _ in range(max_attempts):
        # Format date as string to use with yfinance
        next_date_obj = date_obj + datetime.timedelta(days=1)
        date_str = date_obj.strftime('%Y-%m-%d')
        next_date_str = next_date_obj.strftime('%Y-%m-%d')

        # Suppress warnings from yfinance (((((((NO FUNCIONA LO DE SUPRIMIR LA WARNING)))))))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Check if the market was open on the given date
            data = yf.download('SPY', start=date_str, end=next_date_str, progress=False)  # SPY is a common ticker for market checks
        
        # If data is available, the market was open on 'date_obj'
        if not data.empty:
            return date_str
        
        # If no data is found, move to the next day
        date_obj = next_date_obj

    raise ValueError("No open market date found within the next 10 days.")

def get_adj_close_on_date(tickers, date):

    datee = get_next_open_market_date(date)

    date_obj = datetime.datetime.strptime(datee, '%Y-%m-%d')

    # Get the next day by adding a timedelta of 1 day
    next_date_obj = date_obj + datetime.timedelta(days=1)

    # Download data for the tickers over a single day range to get only that date
    data = None
    # Keep trying until data is downloaded without errors
    while data is None:
        try:
            # Attempt to download data
            data = yf.download(tickers, start=date_obj.strftime('%Y-%m-%d'), end=next_date_obj.strftime('%Y-%m-%d'), group_by='ticker', progress=False)
            # Check if data is empty (in case no data is available for the dates)
            if data.empty:
                raise KeyError("No data available")
        except (KeyError, IndexError):
            # Increment both dates by one day if an error occurs
            date_obj += datetime.timedelta(days=1)
            next_date_obj += datetime.timedelta(days=1)

    # Extract the adjusted close prices
    adj_close_prices = {}
    for ticker in tickers:
        try:
            adj_close_prices[ticker] = data[ticker]['Adj Close'].iloc[0]
        except (KeyError, IndexError):
            adj_close_prices[ticker] = None

    
    return adj_close_prices


tickers = ['AAPL', 'MSFT', 'GOOGL']
date = '2023-02-05'
prices = get_adj_close_on_date(tickers, date)