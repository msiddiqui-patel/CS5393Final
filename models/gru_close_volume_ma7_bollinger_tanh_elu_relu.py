from keras.models import Sequential
from keras.layers import Input, GRU, Dense, ELU, Activation
from keras.optimizers import Adam


def train_gru_close_volume_ma7_bollinger_tanh_elu_relu(
    X_train, y_train, X_test,
    epochs=100, batch_size=32
):
    """
    Build, train, and predict using a GRU model on ['Close', 'Volume_MA_7',
    'Bollinger_Upper', 'Bollinger_Lower'] features, with activations:
    tanh on the first GRU, ELU on the second GRU, and ReLU on the Dense layer.

    Args:
        X_train (np.ndarray): Training input, shape (samples, timesteps, features).
        y_train (np.ndarray): Training targets, shape (samples,).
        X_test  (np.ndarray): Testing input, shape (samples, timesteps, features).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        np.ndarray: Predicted (scaled) values for X_test.
    """
    model = Sequential([
        Input(shape=X_train.shape[1:]),                   # (timesteps, features)
        GRU(50, return_sequences=True, activation='tanh'), # First GRU with tanh
        GRU(50, return_sequences=False, activation='elu'), # Second GRU with ELU
        Dense(25),                                        # Dense before ReLU
        Activation('relu'),                               # ReLU activation
        Dense(1)                                          # Output layer
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=1
    )

    return model.predict(X_test)
