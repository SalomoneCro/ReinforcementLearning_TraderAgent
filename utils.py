from GettingData import get_adj_close_on_date
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
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
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    max_attempts = 10  # Limit to avoid infinite loops in case of an error

    for _ in range(max_attempts):
        # Format date as string to use with yfinance
        next_date_obj = date_obj + timedelta(days=1)
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

def get_adj_close_on_date(tickers: list[str], date: str) -> dict[str, float | None]:
    """
    Retrieves the adjusted close prices for given tickers on a specified date.
    If data for the specified date is unavailable, it increments the date by one day until data is found.

    Parameters:
    - tickers (list[str]): List of stock tickers to retrieve data for.
    - date (str): The date to retrieve data for in 'YYYY-MM-DD' format.

    Returns:
    - dict[str, float | None]: A dictionary with tickers as keys and their adjusted close prices as values.
    """
    datee = get_next_open_market_date(date)
    date_obj = datetime.strptime(datee, '%Y-%m-%d')
    next_date_obj = date_obj + timedelta(days=1)

    data = None
    # Keep trying until data is downloaded without errors
    while data is None:
        try:
            data = yf.download(tickers, start=date_obj.strftime('%Y-%m-%d'), end=next_date_obj.strftime('%Y-%m-%d'), group_by='ticker', progress=False)
            if data.empty:
                raise KeyError("No data available")
        except (KeyError, IndexError):
            date_obj += timedelta(days=1)
            next_date_obj += timedelta(days=1)

    adj_close_prices = {}
    for ticker in tickers:
        try:
            adj_close_prices[ticker] = data[ticker]['Adj Close'].iloc[0]
        except (KeyError, IndexError):
            adj_close_prices[ticker] = None

    return adj_close_prices

def prices_observation(start_date: str, end_date: str, tickers: list[str]) -> dict[str, int]:
    """
    Compares adjusted close prices of tickers between two dates to observe price changes.

    Parameters:
    - start_date (str): The start date for price observation in 'YYYY-MM-DD' format.
    - end_date (str): The end date for price observation in 'YYYY-MM-DD' format.
    - tickers (list[str]): List of stock tickers to observe.

    Returns:
    - dict[str, int]: A dictionary with tickers as keys and values indicating price increase (1) or decrease (-1).
    """
    prices0 = get_adj_close_on_date(tickers=tickers, date=start_date)
    prices1 = get_adj_close_on_date(tickers=tickers, date=end_date)

    comparison_dict = {key: 1 if prices1[key] > prices0[key] else -1 for key in tickers}
    return comparison_dict

def get_stocks(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical stock data for the given tickers from Yahoo Finance.

    Parameters:
    - tickers (list[str]): List of stock tickers to download data for.
    - start_date (str): Start date for the data in 'YYYY-MM-DD' format.
    - end_date (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: A DataFrame containing the adjusted close prices for each ticker.
    """
    stocks_data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    stocks_data.index = stocks_data.index.tz_localize(None)
    return stocks_data

def get_next_week(lista_fechas: list[str], fecha_objetivo: str) -> str:
    """
    Finds the date in a list that is the closest to 7 days in the future of a given date.
    If no date is 7 days ahead in the list, returns the last date in the list.

    Parameters:
    - lista_fechas (list[str]): A list of dates in 'YYYY-MM-DD' format.
    - fecha_objetivo (str): The reference date in 'YYYY-MM-DD' format.

    Returns:
    - str: The closest date to 7 days in the future, or the last date in the list if none is available.
    """
    fechas_dt = [datetime.strptime(fecha, '%Y-%m-%d') for fecha in lista_fechas]
    fecha_objetivo_dt = datetime.strptime(fecha_objetivo, '%Y-%m-%d')
    fecha_limite = fecha_objetivo_dt + timedelta(days=7)
    fechas_futuras = [fecha for fecha in fechas_dt if fecha >= fecha_limite]
    return (min(fechas_futuras) if fechas_futuras else fechas_dt[-1]).strftime('%Y-%m-%d')
