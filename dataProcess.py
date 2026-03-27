import numpy as np
import pandas as pd
import os
import torch
import pickle

class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values.astype(np.float32), axis=-1)
    data_list = [data]
    
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        # 最后一个点之后
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)).astype(np.float32)
        data_list.append(time_in_day)

    """Label Encoding"""
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 1), dtype=np.float32)
        dayofweek_array = df.index.dayofweek.to_numpy(np.float32)
        day_in_week[:, :, 0] = np.tile(dayofweek_array[:, np.newaxis], (1, num_nodes))
        data_list.append(day_in_week)
    
    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, :, 0]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.expand_dims(np.stack(y, axis=0), axis=-1)
    return x, y


def generate_train_val_test(dataset, data_path, batch_size=16, step=12, time_in_day=True, day_in_week=True, train_split=0.7, test_split=0.2):
    """
    Para:
    
    train_split, test_split: Percentage of train/test dataset in total dataset. Sum of then should be less than 1.
    
    """

    if data_path[-4:] == '.csv':
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # df.index = pd.to_datetime(df.index)
    else:
        df = pd.read_hdf(data_path)

    
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.arange(1-step, 1, 1),
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 1+step, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=time_in_day,
        add_day_in_week=day_in_week,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
 
    num_samples = x.shape[0]
    num_test = round(num_samples * test_split)
    num_train = round(num_samples * train_split)
    num_val = num_samples - num_test - num_train
    
    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])
    

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    with open(f'./cache/{dataset}.pkl', 'wb') as file:
        pickle.dump((trainset_loader, valset_loader, testset_loader, scaler), file)
    
    return trainset_loader, valset_loader, testset_loader, scaler



def load_data(dataset, data_path, batch_size=16, step=12, time_in_day=True, day_in_week=True, train_split=0.7, test_split=0.2):
    
    if os.path.exists(f'./cache/{dataset}.pkl'):
        with open(f'./cache/{dataset}.pkl', 'rb') as file:
            trainLoader, valLoader, testLoader, SCALER = pickle.load(file)
    else:
        print("No processed data is founded, now processing the raw data...")
        trainLoader, valLoader, testLoader, SCALER = generate_train_val_test(dataset, data_path, batch_size=batch_size, step=step)

    return trainLoader, valLoader, testLoader, SCALER
    