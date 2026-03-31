import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(ticker, period="5y"):
    """Fetch historical stock data."""
    data = yf.download(ticker, period=period)
    
    # Handle the fact that yfinance returns a MultiIndex when a single ticker is downloaded sometimes, or depending on version.
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(1):
             data = data.xs(ticker, level=1, axis=1)
    
    # Check if 'Close' is there
    if 'Close' not in data.columns:
        # Sometimes ynance returns 'Adj Close', fallback if needed
        return pd.DataFrame() 

    return data[['Close']].copy()

def prepare_data(data, sequence_length=60, future_days=5):
    """
    Prepare data for LSTM.
    Returns:
        X, y, scaler, scaled_data
    """
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - future_days + 1):
        X.append(scaled_data[i - sequence_length:i, 0])
        # predict the next `future_days`.
        y.append(scaled_data[i: i + future_days, 0])

    X, y = np.array(X), np.array(y)
    
    # Reshape X to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, scaled_data
