# Imports
import datetime as dt
import os
from pathlib import Path

# Import pandas
import pandas as pd

# Import yfinance
import yfinance as yf

# Import Auto Regression model
from statsmodels.tsa.ar_model import AutoReg


# Function to fetch stock names and codes from a CSV file
def fetch_stocks():
    # Load the CSV file
    df = pd.read_csv(Path.cwd() / "data" / "equity_issuers.csv")

    # Keep only relevant columns
    df = df[["Security Code", "Issuer Name"]]

    # Create a dictionary mapping
    stock_dict = dict(zip(df["Security Code"], df["Issuer Name"]))

    return stock_dict


# Function to fetch allowed periods and intervals for yfinance
def fetch_periods_intervals():
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }
    return periods


# Function to fetch full stock information
def fetch_stock_info(stock_ticker):
    stock_data = yf.Ticker(stock_ticker)
    info = stock_data.info

    # Safe value getter
    def safe_get(d, key):
        return d.get(key, "N/A")

    stock_info = {
        "Basic Information": {
            "symbol": safe_get(info, "symbol"),
            "longName": safe_get(info, "longName"),
            "currency": safe_get(info, "currency"),
            "exchange": safe_get(info, "exchange"),
        },
        "Market Data": {
            "currentPrice": safe_get(info, "currentPrice"),
            "previousClose": safe_get(info, "previousClose"),
            "open": safe_get(info, "open"),
            "dayLow": safe_get(info, "dayLow"),
            "dayHigh": safe_get(info, "dayHigh"),
            "regularMarketPreviousClose": safe_get(info, "regularMarketPreviousClose"),
            "regularMarketOpen": safe_get(info, "regularMarketOpen"),
            "regularMarketDayLow": safe_get(info, "regularMarketDayLow"),
            "regularMarketDayHigh": safe_get(info, "regularMarketDayHigh"),
            "fiftyTwoWeekLow": safe_get(info, "fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": safe_get(info, "fiftyTwoWeekHigh"),
            "fiftyDayAverage": safe_get(info, "fiftyDayAverage"),
            "twoHundredDayAverage": safe_get(info, "twoHundredDayAverage"),
        },
        "Volume and Shares": {
            "volume": safe_get(info, "volume"),
            "regularMarketVolume": safe_get(info, "regularMarketVolume"),
            "averageVolume": safe_get(info, "averageVolume"),
            "averageVolume10days": safe_get(info, "averageVolume10days"),
            "averageDailyVolume10Day": safe_get(info, "averageDailyVolume10Day"),
            "sharesOutstanding": safe_get(info, "sharesOutstanding"),
            "impliedSharesOutstanding": safe_get(info, "impliedSharesOutstanding"),
            "floatShares": safe_get(info, "floatShares"),
        },
        "Dividends and Yield": {
            "dividendRate": safe_get(info, "dividendRate"),
            "dividendYield": safe_get(info, "dividendYield"),
            "payoutRatio": safe_get(info, "payoutRatio"),
        },
        "Valuation and Ratios": {
            "marketCap": safe_get(info, "marketCap"),
            "enterpriseValue": safe_get(info, "enterpriseValue"),
            "priceToBook": safe_get(info, "priceToBook"),
            "debtToEquity": safe_get(info, "debtToEquity"),
            "grossMargins": safe_get(info, "grossMargins"),
            "profitMargins": safe_get(info, "profitMargins"),
        },
        "Financial Performance": {
            "totalRevenue": safe_get(info, "totalRevenue"),
            "revenuePerShare": safe_get(info, "revenuePerShare"),
            "totalCash": safe_get(info, "totalCash"),
            "totalCashPerShare": safe_get(info, "totalCashPerShare"),
            "totalDebt": safe_get(info, "totalDebt"),
            "earningsGrowth": safe_get(info, "earningsGrowth"),
            "revenueGrowth": safe_get(info, "revenueGrowth"),
            "returnOnAssets": safe_get(info, "returnOnAssets"),
            "returnOnEquity": safe_get(info, "returnOnEquity"),
        },
        "Cash Flow": {
            "freeCashflow": safe_get(info, "freeCashflow"),
            "operatingCashflow": safe_get(info, "operatingCashflow"),
        },
        "Analyst Targets": {
            "targetHighPrice": safe_get(info, "targetHighPrice"),
            "targetLowPrice": safe_get(info, "targetLowPrice"),
            "targetMeanPrice": safe_get(info, "targetMeanPrice"),
            "targetMedianPrice": safe_get(info, "targetMedianPrice"),
        },
    }

    return stock_info


# Function to fetch historical stock data
def fetch_stock_history(stock_ticker, period, interval):
    stock_data = yf.Ticker(stock_ticker)
    history = stock_data.history(period=period, interval=interval)[["Open", "High", "Low", "Close"]]
    return history


# Function to predict stock prices using AutoRegression
def generate_stock_prediction(stock_ticker):
    try:
        stock_data = yf.Ticker(stock_ticker)
        hist = stock_data.history(period="2y", interval="1d")

        # Use only closing prices
        close_prices = hist[["Close"]]

        # Set daily frequency and fill missing values
        close_prices = close_prices.asfreq("D", method="ffill").fillna(method="ffill")

        # Split into training and test sets (90% train)
        split_index = int(len(close_prices) * 0.9)
        train_df = close_prices.iloc[:split_index + 1]
        test_df = close_prices.iloc[split_index:]

        # Train AutoReg model
        model = AutoReg(train_df["Close"], lags=250).fit(cov_type="HC0")

        # Predict for test data
        predictions = model.predict(start=test_df.index[0], end=test_df.index[-1], dynamic=True)

        # Forecast for 90 days into future
        forecast = model.predict(
            start=test_df.index[0],
            end=test_df.index[-1] + dt.timedelta(days=90),
            dynamic=True,
        )

        return train_df, test_df, forecast, predictions

    except Exception as e:
        print("Error in prediction:", e)
        return None, None, None, None
