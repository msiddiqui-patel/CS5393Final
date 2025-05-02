from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam

def train_lstm(X_train, y_train, X_test, epochs=100, batch_size=32):
    """
    Build, train, and predict using an LSTM model.
    """
    model = Sequential([
        # 1) Declare the model’s input signature:
        Input(shape=X_train.shape[1:]),  # (timesteps, features)
        # 2) Now stack LSTM layers without repeating input_shape:
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False, # FORCE CONSISTENCY
        verbose=1
    )

    return model.predict(X_test)