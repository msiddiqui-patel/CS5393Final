import os
import random
import numpy as np
import tensorflow as tf
import argparse
import csv
import time
import json

from data.loader import load_scaled_sequences
from utils.eval    import compute_rmse
# Import model training functions
from models.lstm_model import train_lstm
from models.gru_model  import train_gru
from models.gru_close_tanh_relu_leakyrelu import train_gru_close_tanh_relu_leakyrelu
from models.gru_close_volume_tanh_leakyrelu_elu import train_gru_close_volume_tanh_leakyrelu_elu
from models.gru_close_rsi14_macd_tanh_swish import train_gru_close_rsi14_macd_tanh_swish
from models.gru_close_volume_ma7_bollinger_tanh_elu_relu import train_gru_close_volume_ma7_bollinger_tanh_elu_relu
from models.gru_close_volume_rsi14_macd_bollinger_tanh_swish import train_gru_close_volume_rsi14_macd_bollinger_tanh_swish

# Directory and filenames for outputs
OUTPUT_DIR = 'results'
SEEDED_CSV = os.path.join(OUTPUT_DIR, 'seeded_results.csv')
UNSEEDED_CSV = os.path.join(OUTPUT_DIR, 'unseeded_results.csv')
SEEDED_JSON = os.path.join(OUTPUT_DIR, 'seeded_results.json')
UNSEEDED_JSON = os.path.join(OUTPUT_DIR, 'unseeded_results.json')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model registry
defined_models = {
    'LSTM': train_lstm,
    'GRU': train_gru,
    'GRU_CLOSE_TANH_RELU_LEAKYRELU': train_gru_close_tanh_relu_leakyrelu,
    'GRU_CLOSE_VOLUME_TANH_LEAKYRELU_ELU': train_gru_close_volume_tanh_leakyrelu_elu,
    'GRU_CLOSE_RSI14_MACD_TANH_SWISH': train_gru_close_rsi14_macd_tanh_swish,
    'GRU_CLOSE_VOLUME_MA7_BOLLINGER_TANH_ELU_RELU': train_gru_close_volume_ma7_bollinger_tanh_elu_relu,
    'GRU_CLOSE_VOLUME_RSI14_MACD_BOLLINGER_TANH_SWISH': train_gru_close_volume_rsi14_macd_bollinger_tanh_swish
}

# Functions to seed and unseed

def seed_all(seed=42):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate RNN models with seeded and unseeded runs"
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        '--batch_size', '-b', type=int, default=32,
        help="Batch size for training (default: 32)"
    )
    args = parser.parse_args()

    # Load data once
    X_train, y_train, X_test, y_test, scaler = load_scaled_sequences(
        ticker="AAPL", start="2010-01-01", end="2023-11-13",
        seq_length=60, train_split=0.8
    )

    # Seeded single-run results
    seed_all(42)
    seeded_results = []
    with open(SEEDED_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'rmse', 'time_sec'])
        for name, fn in defined_models.items():
            start = time.time()
            preds = fn(X_train, y_train, X_test,
                       epochs=args.epochs,
                       batch_size=args.batch_size)
            elapsed = time.time() - start
            rmse = compute_rmse(preds, y_test, scaler)
            writer.writerow([name, f"{rmse:.6f}", f"{elapsed:.2f}"])
            seeded_results.append({
                'model': name,
                'rmse': float(f"{rmse:.6f}"),
                'time_sec': float(f"{elapsed:.2f}")
            })
    # Write JSON
    with open(SEEDED_JSON, 'w') as f:
        json.dump(seeded_results, f, indent=4)
    print(f"Seeded results written to {SEEDED_CSV} and {SEEDED_JSON}")

    # Unseeded three-run results
    if 'TF_DETERMINISTIC_OPS' in os.environ:
        del os.environ['TF_DETERMINISTIC_OPS']
    unseeded_results = []
    with open(UNSEEDED_CSV, 'w', newline='') as f:
        header = ['model']
        for run in range(1, 4):
            header += [f'run{run}_rmse', f'run{run}_time_sec']
        header += ['avg_rmse', 'avg_time_sec']
        writer = csv.writer(f)
        writer.writerow(header)

        for name, fn in defined_models.items():
            runs = []
            for run in range(3):
                start = time.time()
                preds = fn(X_train, y_train, X_test,
                           epochs=args.epochs,
                           batch_size=args.batch_size)
                elapsed = time.time() - start
                rmse = compute_rmse(preds, y_test, scaler)
                runs.append({'rmse': float(f"{rmse:.6f}"), 'time_sec': float(f"{elapsed:.2f}")})
            avg_rmse = sum(r['rmse'] for r in runs) / 3
            avg_time = sum(r['time_sec'] for r in runs) / 3
            row = [name]
            for r in runs:
                row += [f"{r['rmse']:.6f}", f"{r['time_sec']:.2f}"]
            row += [f"{avg_rmse:.6f}", f"{avg_time:.2f}"]
            writer.writerow(row)
            unseeded_results.append({
                'model': name,
                'runs': runs,
                'avg_rmse': float(f"{avg_rmse:.6f}"),
                'avg_time_sec': float(f"{avg_time:.2f}")
            })
    # Write JSON
    with open(UNSEEDED_JSON, 'w') as f:
        json.dump(unseeded_results, f, indent=4)
    print(f"Unseeded results written to {UNSEEDED_CSV} and {UNSEEDED_JSON}")

if __name__ == '__main__':
    main()
