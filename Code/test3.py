from finta import TA
import pandas as pd

import numpy as np
np.random.seed(42)
date_range = pd.date_range(start='2025-01-01 9:15:00', periods=100, freq='H')
print(date_range)
open_price = np.random.uniform(100,200,size=100)

x={"close": [100, 102, 101, 103, 105],"open": [100, 102, 101, 103, 105],"high": [100, 102, 101, 103, 105],"low": [100, 102, 101, 103, 105]}

df = pd.DataFrame(x)
df["RSI"] = TA.RSI(df)
print(df)
