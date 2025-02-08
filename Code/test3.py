from finta import TA
import pandas as pd
x={"close": [100, 102, 101, 103, 105],"open": [100, 102, 101, 103, 105],"high": [100, 102, 101, 103, 105],"low": [100, 102, 101, 103, 105]}

df = pd.DataFrame(x)
df["RSI"] = TA.RSI(df)
print(df)
