from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd


def generate_graph_seq2seq_io_data(
    df: pd.DataFrame,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    signal_lengths: list,
    cycle_num: int,
    add_time_in_day: bool = False,
    add_day_in_week: bool = False,
    scaler=None,
) -> tuple:
    """
    Generate input-output sequence data for graph-based sequence-to-sequence modeling.

    Args:
        df: DataFrame with time series data (rows: time, cols: nodes).
        x_offsets: Array of offsets for input sequences (e.g., [-19, -18, ..., 0]).
        y_offsets: Array of offsets for output sequences (e.g., [1, 2, ..., 6]).
        signal_lengths: List of signal cycle lengths for each node.
        cycle_num: Number of past cycles to include.
        add_time_in_day: Whether to include time-of-day as a feature.
        add_day_in_week: Whether to include day-of-week as a feature.
        scaler: Optional scaler for data normalization (not implemented).

    Returns:
        Tuple (x, y) where:
            - x: Input data with shape (num_samples, input_length, num_nodes, input_dim).
            - y: Target data with shape (num_samples, output_length, num_nodes, output_dim).
    """
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)  # Shape: (num_samples, num_nodes, 1)
    seq_input = x_offsets.shape[0]
    feature_list = [data]
    interval = df.index[1] - df.index[0]  # Time interval between consecutive samples
    print("Sample interval:", interval)

    # Add time-of-day feature
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    # Add day-of-week feature
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    # Add cycle index for each timestamp
    if args.add_signal_index:
        signal_index = np.zeros((num_samples, seq_input, num_nodes, 1))
        for i in range(num_nodes):
            for j in range(num_samples):
                time_diff_ns = df.index.values[j] - df.index.values[0]
                time_diff_seconds = time_diff_ns.astype("int64") / 1e9
                signal_index[j][0][i][0] = time_diff_seconds % signal_lengths[i]

    # Add data from past cycles at the same position
    if args.add_past_cycle:
        pastcycle_data = np.zeros((num_samples, seq_input, num_nodes, 1))
        for i in range(num_nodes):
            for j in range(num_samples):
                default = df.values[j, i]  # Use current value as fallback
                for k in range(seq_input):
                    if k < cycle_num:
                        cycle = signal_lengths[i]
                        past_time = df.index[j] - (k + 1) * cycle
                        if past_time >= df.index[0]:  # Check if past timestamp exists
                            index = j - int((k + 1) * cycle / interval)
                            pastcycle_data[j][k][i][0] = df.values[index, i]
                        else:
                            pastcycle_data[j][k][i][0] = default

    # Combine all features
    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))  # Start index for sampling
    max_t = num_samples - abs(max(y_offsets))  # End index for sampling

    # Generate input-output pairs
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)  # Shape: (num_samples, seq_input, num_nodes, input_dim)
    y = np.stack(y, axis=0)  # Shape: (num_samples, seq_output, num_nodes, output_dim)

    # Combine additional features if enabled
    if args.add_past_cycle and args.add_signal_index:
        x_cat = np.concatenate([x, pastcycle_data[min_t:max_t], signal_index[min_t:max_t]], axis=-1)
        return x_cat, y
    return x, y


def generate_train_val_test(args):
    """Generate and split data into training, validation, and test sets."""
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = pd.read_csv(args.dataset, index_col=0, parse_dates=True)
    print("DataFrame:\n", df)

    # Define input and output offsets
    x_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1))
    y_offsets = np.sort(np.arange(args.y_start, seq_length_y + 1, 1))

    # Load signal lengths from CSV
    with open(args.signal_length, mode="r", newline="") as file:
        reader = csv.reader(file)
        signal_lengths = [int(x) for row in reader for x in row]

    # Generate data
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        signal_lengths=signal_lengths,
        cycle_num=args.cycle_num,
        add_time_in_day=False,
        add_day_in_week=args.dow,
    )

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")

    # Split into train, val, test sets
    num_samples = x.shape[0]
    num_val = round(num_samples * 0.1)
    num_train = round(num_samples * 0.7)
    num_test = num_samples - num_val - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]

    # Save splits to disk
    for cat in ["train", "val", "test"]:
        _x, _y = locals()[f"x_{cat}"], locals()[f"y_{cat}"]
        print(f"{cat} - x: {_x.shape}, y: {_y.shape}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def str_to_bool(value):
    """Convert a string to a boolean value."""
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    if value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sequence data for graph-based models.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/output_file.csv",
        help="Path to the input dataset CSV file",
    )
    parser.add_argument("--seq_length_x", type=int, default=20, help="Input sequence length")
    parser.add_argument("--seq_length_y", type=int, default=6, help="Output sequence length")
    parser.add_argument("--y_start", type=int, default=1, help="Starting index for prediction")
    parser.add_argument("--dow", action="store_true", help="Include day-of-week feature")
    parser.add_argument(
        "--signal_length",
        type=str,
        default="data/signal.csv",
        help="Path to the signal length CSV file",
    )
    parser.add_argument("--cycle_num", type=int, default=5, help="Number of past cycles to include")
    parser.add_argument(
        "--add_past_cycle",
        type=str_to_bool,
        default=True,
        help="Include past cycle data",
    )
    parser.add_argument(
        "--add_signal_index",
        type=str_to_bool,
        default=True,
        help="Include signal index",
    )

    args = parser.parse_args()
    print("Arguments:", args)

    # Set output directory based on feature flags
    args.output_dir = "data/DATA" if (args.add_past_cycle and args.add_signal_index) else "data/DATA_noSignal"

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate and save the data
    generate_train_val_test(args)