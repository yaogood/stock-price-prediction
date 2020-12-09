import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset

from utils.chart import plot_candlestick, plot_finance, plot_timeseries


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.Tensor(self.x[idx, :, :]), torch.Tensor(self.y[idx])


def normalize(x, y, y_scaler=None):
    x = x.copy()
    y = y.copy()

    for feature_i in range(x.shape[1]):
        train_values = x[:, feature_i]
        scaler = preprocessing.MinMaxScaler(feature_range=[0, 1])
        train_values = scaler.fit_transform(train_values.reshape(-1, 1))
        x[:, feature_i] = train_values.flatten()

    if y_scaler:
        return x, y_scaler.fit_transform(y)
    else:
        return x, y


def load_data(file_path, sheet_name, x_w_size=30, y_w_size=5, if_normalize_y=False):

    df = pd.read_excel(file_path, sheet_name) 
    df.columns= df.columns.str.lower()
    df.columns = [c.replace(' ', '_') for c in df.columns]
    df = df.rename(
        columns={'close_price':'close', 'open_price':'open', 'high_price':'high', 'low_price':'low', 'ntime':'date'})
    df['date'] = pd.to_datetime(df.date.astype(str), format='%Y%m%d')
    
    # print(df.columns)
    # plot_finance(df, './images/')

    fval = df.at[0, 'close']
    df['change'] = (df['close'] / df['close'].shift(periods=1, fill_value=fval) - 1) * 100

    assert not df.isnull().values.any()

    
    data_x = df.drop(['date', 'time', 'federal_fund_rate'], axis=1).values
    data_y = df['change'].astype(float).values.reshape(-1, 1)
    data_x, data_y = normalize(data_x, data_y)

    date = df['date'].values.reshape(-1, 1)
    close = df['close'].astype(float).values.reshape(-1, 1)

    # print(data_x.shape, data_y.shape)
    # print(data_x[0], '\n', data_y[0])

    total_w_size = x_w_size + y_w_size
    dataset_length = data_x.shape[0] - total_w_size

    src = np.empty([dataset_length, x_w_size, data_x.shape[1]])   # [num_instances, sequence_len, src_vector]
    tgt = np.empty([dataset_length, y_w_size, 1])                   # [num_instances, sequence_len, 1]

    date_time = np.empty([dataset_length, y_w_size, 1], dtype='datetime64[D]')
    close_price = np.empty([dataset_length, y_w_size, 1])

    for i in range(dataset_length):
        src[i, :, :] = data_x[i:i+x_w_size, :]
        tgt[i, :, :] = data_y[i+x_w_size: i+total_w_size]

        date_time[i, :, :] = date[i+x_w_size: i+total_w_size]
        close_price[i, :, :] = close[i+x_w_size: i+total_w_size]

    train_valid_len = int(0.9*src.shape[0])
    train_len = int(0.89*train_valid_len)
    tv_slice = np.random.permutation(train_valid_len)

    x = src[tv_slice[:train_len], :, :]
    y = tgt[tv_slice[:train_len], :, :]

    x_val = src[tv_slice[train_len:], :, :]
    y_val = tgt[tv_slice[train_len:], :, :]
    
    x_test = src[train_valid_len:, :, :]
    y_test = tgt[train_valid_len:, :, :]

    date_time = date_time[train_valid_len:, :, :]
    close_price = close_price[train_valid_len:, :, :]

    print(src.shape, tgt.shape, x.shape, y.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape, date_time.shape, close_price.shape)

    # print(x[:5],'\n', y[:5],'\n', x_val[:5],'\n', y_val[:5],'\n', x_test[:5],'\n', y_test[:5],'\n', date_time[:5],'\n', close_price[:5])

    return x, y, x_val, y_val, x_test, y_test, date_time, close_price



# x, y, x_val, y_val, x_test, y_test, date, close_price_true = \
#         load_data('./raw_data.xlsx', 'S&P500 Index Data', 30, 1)