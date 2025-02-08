import numpy as np
import pandas as pd
import pandas_ta as ta  # Technical analysis library
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load stock data
data = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)

# Compute technical indicators using pandas-ta
data['SMA_20'] = ta.sma(data['Close'], length=20)  # Simple Moving Average (20 days)
data['EMA_10'] = ta.ema(data['Close'], length=10)  # Exponential Moving Average (10 days)
data['RSI'] = ta.rsi(data['Close'], length=14)  # Relative Strength Index
data['MACD'], data['MACD_Signal'], _ = ta.macd(data['Close'])  # MACD Indicator
data.dropna(inplace=True)  # Drop rows with NaN values

# Select features
features = ['Close', 'SMA_20', 'EMA_10', 'RSI', 'MACD', 'MACD_Signal']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])  # Predicting the closing price
    return np.array(X), np.array(y)

seq_length = 60  # Using last 60 days of data
X, y = create_sequences(scaled_data, seq_length)

# Train-test split (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, len(features))),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Predicting the stock closing price
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.hstack([predictions, np.zeros((predictions.shape[0], len(features)-1))]))[:, 0]  # Rescale

# Convert actual values back
y_test_rescaled = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(features)-1))]))[:, 0]

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction Using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
