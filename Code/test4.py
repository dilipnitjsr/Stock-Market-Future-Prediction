import vectorbt as vbt
import pandas as pd
import pandas_ta as ta

# Sample Data
df = pd.DataFrame({"close": [100, 102, 101, 103, 105]})

# Apply RSI indicator
df["RSI"] = ta.rsi(df["close"], length=14)
print(df)
# Sample Price Data
prices = [100, 102, 101, 103, 105]

# Compute RSI
rsi = vbt.IndicatorFactory.from_pandas(pd.Series(prices)).rsi()
print(rsi)