from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import csv
from datetime import datetime


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, signal_lengths, cycle_num, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples for sequence-to-sequence model training.
    
    :param df: Input DataFrame with time series data.
    :param x_offsets: Offsets for input sequences.
    :param y_offsets: Offsets for output sequences.
    :param signal_lengths: List of signal lengths for each node.
    :param cycle_num: Number of past cycles to consider.
    :param add_time_in_day: Whether to add time of day as a feature.
    :param add_day_in_week: Whether to add day of week as a feature.
    :param scaler: Optional scaler for data normalization.
    :return: Tuple of (x, y) where x is the input data and y is the target data.
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    seq_input = x_offsets.shape[0]
    # print("data.shape", data.shape)  # (17240, 361, 1)
    feature_list = [data]
    interval = df.index[1] - df.index[0]  # 相邻时间点之间间隔
    print("sample interval", interval)

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    if args.add_signal_index:  # add cycle index for each time stamp
        signal_index = np.zeros((num_samples, seq_input, num_nodes, 1))
        for i in range(num_nodes):
          for j in range(num_samples):
            time_diff_ns = df.index.values[j] - df.index.values[0]
            time_diff_seconds = time_diff_ns.astype('int64') / 1e9
            signal_index[j][0][i][0] = time_diff_seconds % signal_lengths[i]
            
    if args.add_past_cycle:  # add data of the same position in k past cycle
        pastcycle_data = np.zeros((num_samples, seq_input, num_nodes, 1))
        for i in range(num_nodes):
          for j in range(num_samples):
            default = df.index.values[j]
            for k in range(seq_input):
              if k < cycle_num:
                cycle = signal_lengths[i]
                if df.index.values[j] - (k + 1) * cycle >= df.index.values[0]:  # the timestamp exists
                  index = j - int((k + 1) * cycle / interval)
                  default = df.index.values[index]
                  pastcycle_data[j][k][i][0] = default
                else:  # not exists, use the last cycle data
                  pastcycle_data[j][k][i][0] = default
    
    data = np.concatenate(feature_list, axis=-1)  # (17240, 361, 1)
    # print("data.shape", data.shape)
    x, y = [], []
    min_t = abs(min(x_offsets))  # 19
    max_t = abs(num_samples - abs(max(y_offsets)))  # 17220
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    if args.add_past_cycle and args.add_signal_index:
      x_cat = np.concatenate([x, pastcycle_data[min_t:max_t], signal_index[min_t:max_t]], axis=-1)
      return x_cat, y
    else:
      return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = pd.read_csv(args.dataset, index_col=0, parse_dates=True)
    print("df\n", df)
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    signal = []
    with open(args.signal_length, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            signal.extend(row)

    signal_lengths = [int(x) for x in signal]
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        signal_lengths=signal_lengths,
        cycle_num=args.cycle_num,
        add_time_in_day=False,
        add_day_in_week=args.dow,
    )

    print("x shape: ", x.shape)
    print("y shape: ", y.shape)
    
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_val = round(num_samples * 0.1)
    num_train = round(num_samples * 0.7)
    num_test = num_samples - num_val - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/output_file.csv',help='data path')
    parser.add_argument("--seq_length_x", type=int, default=20, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=6, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)
    parser.add_argument("--signal_length", type=str, default='data/signal.csv',help='signal length data path')
    parser.add_argument("--cycle_num", type=int, default=5, help="past k cycle data of the same position")
    parser.add_argument("--add_past_cycle", type=str_to_bool, default=True, help='')
    parser.add_argument("--add_signal_index", type=str_to_bool, default=True, help='')
    args = parser.parse_args()
    if args.add_past_cycle and args.add_signal_index:
       args.output_dir = "data/DATA"
    else:
       args.output_dir = "data/DATA_noSignal"
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
