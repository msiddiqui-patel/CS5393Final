from keras.models import Sequential
from keras.layers import Input, GRU, Dense
from keras.optimizers import Adam


def train_gru(X_train, y_train, X_test, epochs=100, batch_size=32):
    """
    Build, train, and predict using a GRU model.

    Args:
        X_train (np.ndarray): Training input data, shape (samples, timesteps, features).
        y_train (np.ndarray): Training targets, shape (samples,).
        X_test  (np.ndarray): Testing input data, shape (samples, timesteps, features).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        np.ndarray: Predicted values for X_test (scaled).
    """
    # Define model architecture with explicit Input layer
    model = Sequential([
        Input(shape=X_train.shape[1:]),  # (timesteps, features)
        GRU(50, return_sequences=True),
        GRU(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False, # FORCE CONSISTENCY
        verbose=1
    )

    # Predict and return
    return model.predict(X_test)
