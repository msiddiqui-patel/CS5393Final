import numpy as np


def compute_rmse(predictions, y_true, scaler):
    # predictions: array of shape (n_samples, 1) in scaled space
    # y_true:      array of shape (n_samples,) in scaled space

    # Invert scaling
    inv_pred = scaler.inverse_transform(
        np.concatenate([predictions, np.zeros_like(predictions)], axis=1)
    )[:, 0]

    inv_true = scaler.inverse_transform(
        np.concatenate([y_true.reshape(-1, 1), np.zeros_like(y_true.reshape(-1, 1))], axis=1)
    )[:, 0]

    # RMSE
    return np.sqrt(np.mean((inv_pred - inv_true) ** 2))