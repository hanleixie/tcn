import torch
import torch.nn as nn
import torch.nn.functional as function
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from pandas.core.frame import DataFrame
import logging as log
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # 日志格式化输出
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"  # 日期格式
fp = log.FileHandler('ser_Bilstm_log.txt', encoding='utf-8')
fs = log.StreamHandler()
log.basicConfig(level=log.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fp, fs])
# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
father_path_1 = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
father_path_2 = os.path.abspath(os.path.dirname(father_path_1) + os.path.sep + ".")
sys.path.append(father_path_2)
# 设置输出编码为utf-8  防止乱码
utf8_stdout = os.fdopen(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
sys.stdout = utf8_stdout
import time
import warnings
warnings.filterwarnings("ignore")
import argparse
from TCN.adding_problem.model import TCN

parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false', default=False,
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=4,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=7,
                    help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=400,
                    help='sequence length (default: 400)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()


csv_path = r'C:\Users\Administrator\PycharmProjects\one\lizard-reday\data\data-STOCKBILL3128_ZZ-fuzhou2.csv'
df = pd.read_csv(csv_path)
#date_index = pd.to_datetime(df['ACCTDATE'], format='%Y%m%d')
#df['ACCTDATE'] = date_index
date = df.iloc[:, 0].values
data = df.iloc[:, -1].values
df = pd.DataFrame({'date_id': date, 'data_id': data})

dt_format = '%Y%m%d'
date_type = 'D'

date_index = pd.to_datetime(df['date_id'], format=dt_format)
df['date_id'] = date_index

index_all = pd.date_range(str(df['date_id'].iloc[0]),  # periods=len(date_index), freq=date_type)
                          end=str(df['date_id'].iloc[-1]), freq=date_type)
if date_type == 'M':
    index_all = pd.date_range(str(df['date_id'].iloc[0]), periods=len(index_all) + 1, freq=date_type)

if len(df) == len(index_all):
    original_data = df
else:
    date_df = pd.DataFrame(index_all, columns=['date_id'])
    original_data = pd.merge(date_df, df, on='date_id', how='left')
    original_data = original_data.fillna(method='bfill')
#data_id = df['data_id'].apply(lambda x: np.log10(x)).values
data = list(original_data['data_id'].apply(lambda x: np.log10(x)).values)

def data_generate(data, step_back=1, train_ratio=0.7):
    #data = np.asarray(data).reshape((len(data), 1))
    #scaler = MinMaxScaler()#feature_range=(0, 1))
    #data = scaler.fit_transform(data)
    x_data = []
    y_data = []
    for i in range(len(data)-step_back):
        x_data.append(data[i:i+step_back])
        y_data.append(data[i+step_back])

    '''
    
    x_train, x_test, y_train, y_test = train_test_split(dataSet, labels, test_size=0.3, random_state=42)
    print( x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    out:
    
    ((51527, 500), (51527,), (22084, 500), (22084,))
    1.3 数据的结构
    1.3.1 将 numpy 数据集转化为 tensor
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    1.3.2 训练集数据类型转化为：tensor.float32
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    1.3.3 改变 x_train.shape , x_test.shape 的形状
    送入训练的数据格式为：（1， 1， 500）
    
    x_train = x_train.reshape(x_train.shape[0], 1, 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_train.shape[1])
    print(x_train.shape, x_test.shape)
    out:
    
    (torch.Size([51527, 1, 1, 500]), torch.Size([22084, 1, 500]))
    1.3.4 标签的数据格式 与 数据类型
    在1.2 节中给出标签的形状：x_train.shape = (51527,) , y_train.shape = (22084,)，，类型是：float， 但是训练器需要的是：（51527, 1）, (22084, 1) ,dtype=tensor.long

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    '''

    x_data = np.asarray(x_data)[:800]
    y_data = np.asarray(y_data)[:800]
    print('11', x_data.shape)#[877,2]
    print('222', y_data.shape)#[877,]
    train_size = int(len(x_data)*train_ratio)
    print(train_size)

    #x_train = x_data[:train_size].view(-1, 1, step_back)
    #y_train = y_data[:train_size].view(-1, 1)
    x_train = x_data[:int(len(x_data)*train_ratio)].reshape(-1, 1, step_back)
    y_train = y_data[:int(len(x_data)*train_ratio)].reshape(-1, 1)

    #x_ = x_data[:train_size].view(-1, 1, step_back)
    #y_train = y_data[:train_size].view(-1, 1)
    x_vail = x_data[int(len(x_data)*train_ratio):].reshape(-1, 1, step_back)
    y_vail = y_data[int(len(x_data)*train_ratio):].reshape(-1, 1)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_vail = torch.from_numpy(x_vail)
    y_vail = torch.from_numpy(y_vail)
    x_train = Variable(x_train).type(torch.FloatTensor)
    y_train = Variable(y_train).type(torch.FloatTensor)
    x_vail = Variable(x_vail).type(torch.FloatTensor)
    y_vail = Variable(y_vail).type(torch.FloatTensor)

    return x_train, y_train, x_vail, y_vail

x_train, y_train, x_vail, y_vail = data_generate(data, step_back=3, train_ratio=0.7)
print('1:', x_train.shape)
print('2:', y_vail.shape)


input_channels = 1
n_classes = 1
batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs
channel_sizes = [args.nhid]*args.levels
print('channel_sizes', channel_sizes)
kernel_size = 2#args.ksize
dropout = args.dropout
#input_channels=1, n_classes=1, channel_sizes=[30, 30, 30, 30, 30, 30, 30, 30], kernel_size=7, dropout=0
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

lr = args.lr
optimizer = getattr(torch.optim, args.optim)(model.parameters(), lr=lr)

def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, x_train.size(0), batch_size):
        if i + batch_size > x_train.size(0):

            x, y = x_train[i:], y_train[i:]
            print('3', x_train[i:])
        else:
            x, y = x_train[i:(i+batch_size)], y_train[i:(i+batch_size)]
        optimizer.zero_grad()
        output = model(x)
        loss = function.mse_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, x_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, x_train.size(0), 100.*processed/x_train.size(0), lr, cur_loss))
            total_loss = 0

def fangui(data):
    data = np.asarray(data).reshape((len(data), 1))
    scaler = MinMaxScaler()  # feature_range=(0, 1))
    # scaler = scaler.fit(data)
    # 归一化数据集并打印前5行
    #data = scaler.fit_transform(data)
    data = scaler.inverse_transform(data)
    return data

def evaluate():
    model.eval()
    with torch.no_grad():
        output = model(x_vail)
        test_loss = function.mse_loss(output, y_vail)

        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
        return test_loss.item(), output, y_vail

loss = []
for ep in range(1, epochs+1):
    train(ep)
    tloss, output, y_vail = evaluate()
    loss.append(tloss)
output = [10**x for x in output]
y_vail = [10**x for x in y_vail]
plt.figure(figsize=(8, 8), dpi=80)
plt.plot(output, label="fore_data", color='r')
plt.plot(y_vail, label="raw_data", color='b')
plt.legend(loc=0)
plt.tick_params(axis='x', labelsize=8)
plt.xticks(rotation=-90)
plt.title("Comparison of raw data and forecast data")
plt.figure(figsize=(8, 8), dpi=80)
plt.plot(loss, label="original", color='r')
plt.title("pred_ave_loss")
plt.xticks(rotation=-90)
plt.show()

#print(model)
















