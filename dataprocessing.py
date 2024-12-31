import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker="AAPL", start_date="2018-01-01", end_date="2023-01-01"):
    """
    Loads data from yfinance and returns a pandas DataFrame.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']]  # Keep it simple: only 'Close' price
    df.dropna(inplace=True)
    return df

def create_sequences(data, sequence_length=60):
    """
    Scales data, then creates sequences of length `sequence_length`.
    Returns the scaled data, plus (X, y) for training/testing.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshape for LSTM: (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler
