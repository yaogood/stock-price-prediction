from math import sqrt
import os
import time
import dateutil
import logging
import numpy as np
import torch
import shutil
from datetime import datetime

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def create_exp_path(root_exp_dir):
    if not os.path.exists(root_exp_dir):
        os.makedirs(root_exp_dir)
    # set log path
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    cur_exp_dir = os.path.join(root_exp_dir, timestamp)
    os.makedirs(cur_exp_dir)

    # set checkpoint path
    ckpt_path = os.path.join(cur_exp_dir, 'model')
    os.makedirs(ckpt_path)

    log_path = os.path.join(cur_exp_dir, 'log')
    os.makedirs(log_path)

    return log_path, ckpt_path


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)

    logging.basicConfig(filename=str(final_log_file),
                        level=logging.INFO,
                        format='%(asctime)-5s %(message)s')

    logger = logging.getLogger()
    console = logging.StreamHandler()
    logger.addHandler(console)
    return logger


def save_all(model_path, args, model, optimizer, epoch, is_best=False):
    state_dict = {
        'args': args,
        'model': model.state_dict() if model else {},
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    filename = os.path.join(model_path, 'checkpoint{}.pt'.format(epoch))
    torch.save(state_dict, filename)
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    shutil.copyfile(filename, newest_filename)
    if is_best:
        best_filename = os.path.join(model_path, 'checkpoint_best.pt')
        shutil.copyfile(filename, best_filename)


def save(model_path, args, model, optimizer, epoch, is_best=False):
    state_dict = {
        'args': args,
        'model': model.state_dict() if model else {},
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    filename = os.path.join(model_path, 'checkpoint.pt')
    torch.save(state_dict, filename)
    if is_best:
        best_filename = os.path.join(model_path, 'checkpoint_best.pt')
        shutil.copyfile(filename, best_filename)


# sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# sklearn.metrics.r2_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
# scipy.stats.pearsonr
# def R_score(y_true, y_pred):
#     y_true_mean = np.mean(y_true)
#     y_pred_mean = np.mean(y_pred)

#     return np.sum( (y_true - y_true_mean) * (y_pred - y_pred_mean) ) \
#         / np.sqrt( np.sum(((y_true - y_true_mean)**2) * ((y_pred - y_pred_mean)**2)) )

def R_score(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)

    return np.mean( (y_true - y_true_mean) * (y_pred - y_pred_mean) ) \
        / np.sqrt( np.mean((y_true - y_true_mean)**2) * np.mean((y_pred - y_pred_mean)**2) )

def Theil_U(y_true, y_pred):
    a = np.sqrt(np.mean((y_true - y_pred)**2))
    b = np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2))
    return a/b

def measure_all(y_pred, date, close_price_true):
    y_pred = np.stack(y_pred).flatten()
    date = date.flatten()
    close_price_true = close_price_true.flatten()

    assert y_pred.shape == date.shape == close_price_true.shape

    close_price_pred = np.empty(y_pred.shape)
    
    for i in range(1, y_pred.shape[0]):
        close_price_pred[i] = close_price_true[i-1] * (1 + y_pred[i]/100)
    # print(close_price_pred)
    # print(close_price_true) 
    close_price_true[np.where(close_price_true == 0)] = 0.0000001
    close_price_pred[np.where(close_price_pred == 0)] = 0.0000001

    mape_score  = MAPE(close_price_true[1:], close_price_pred[1:])
    r_score     = R_score(close_price_true[1:], close_price_pred[1:])
    theil_score = Theil_U(close_price_true[1:], close_price_pred[1:])

    return mape_score, r_score, theil_score