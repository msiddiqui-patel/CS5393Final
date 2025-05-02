import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def load_scaled_sequences(ticker, start, end, seq_length, train_split=0.8):
    # Download
    df = yf.download(
        ticker,
        start = start,
        end = end,
        auto_adjust = True  # or False if you really want raw closes
    )[['Close']].dropna()

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    # Build sequences
    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i: i + seq_length, 0])
        y.append(scaled[i + seq_length, 0])
    X, y = np.array(X), np.array(y)

    # Train/test split
    split = int(len(X) * train_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for RNN: (samples, timesteps, features)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, y_train, X_test, y_test, scaler