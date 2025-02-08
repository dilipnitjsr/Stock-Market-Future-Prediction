import yfinance as yf
import pandas as pd

ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2025-01-01")
data.to_csv("stock_data.csv")  # Save for reusedir