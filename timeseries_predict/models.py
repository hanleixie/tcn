# -*- coding:utf-8 -*-
# @Time: 2020/10/22 15:34
# @File: models.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
from Data_Preprocess import *
from config import *
from log import logger, init_logger
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import json
from tqdm import *
import time
import matplotlib.pyplot as plt

from fbprophet import Prophet
class Prophetor_modle():
    def __init__(self, num_preds=10, date_format='M'):
        '''
        :param num_preds：预测时间长度
        :param date_format：日期类型
        :param
        '''
        self.num_preds = num_preds
        self.freq = self.format_type(date_format=date_format)

    def format_type(self, date_format='%Y%m%d'):
        if date_format == '%Y':
            self.freq = 'Y'
        elif date_format == '%Y%m':
            self.freq = 'M'
        else:
            self.freq = 'D'

    def fitAndPred(self, train_data,
                   yearly_seasonality='auto',
                   weekly_seasonality=False,
                   seasonality_mode='additive',  # multiplicative | additive
                   changepoint_prior_scale=0.04,  # default: 0.05
                   changepoint_range=0.85,  # default: 0.8
                   seasonality_prior_scale=10.0,  # default: 10.0
                   growth='linear',
                   interval_width=0.80,
                   changepoints=None,
                   monthly_seasonality=False):
        # train_data['floor'] = floor
        # train_data['cap'] = cap
        pf_model = Prophet(yearly_seasonality=yearly_seasonality,
                           weekly_seasonality=weekly_seasonality,
                           seasonality_mode=seasonality_mode,  # multiplicative | additive
                           changepoint_prior_scale=changepoint_prior_scale,  # default: 0.05
                           changepoint_range=changepoint_range,  # default: 0.8
                           seasonality_prior_scale=seasonality_prior_scale,  # default: 10.0
                           growth=growth,
                           interval_width=interval_width,
                           changepoints=changepoints)  # , mcmc_samples=300)

        dict_ = {'ds': train_data.index, 'y': train_data.values}
        _train_data = pd.DataFrame(dict_)
        if monthly_seasonality:
            pf_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        if growth == 'logistic':
            _train_data['cap'] = max(_train_data[['y']].values)[0]
            #pf_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            pf_model.fit(_train_data)
            _future = pf_model.make_future_dataframe(periods=self.num_preds, freq=self.freq)
            _future['cap'] = max(_train_data[['y']].values)[0]
            _forecast = pf_model.predict(_future)
        else:
            pf_model.fit(_train_data)
            _future = pf_model.make_future_dataframe(periods=self.num_preds, freq=self.freq)
            _forecast = pf_model.predict(_future)
            forecast = _forecast['yhat'].values
        #print(forecast[[column_ds, 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))
        return forecast

# lightgbm model
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
class Lightgbm_model():
    def __init__(self, ratio=0.1, date_format='%Y%m%d'):
        '''
        :param ratio：测试集划分比例
        :param date_format：日期格式
        '''
        self.ratio = ratio
        self.date_format = date_format

    def predict(self, new_week_feature):

        train_data, train_label = new_week_feature[:, :-2], new_week_feature[:, -2]

        x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=self.ratio)
        lgb_train = lgb.Dataset(x_train, label=y_train)
        lgb_eval = lgb.Dataset(x_test, label=y_test)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',  # {'l1', 'l2'},
            'num_leaves': 300,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 4,
            'verbose': 0,
            'max_depth': 3
        }
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=200,
                        valid_sets=lgb_eval,
                        # verbose_eval=200,
                        early_stopping_rounds=200,
                        )
        y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)

        return y_pred, y_test

    def predict_all(self, new_week_feature, pred_date_feature, last_year_month):
        from datetime import datetime, timedelta
        global future_true_data, pred_data
        date = new_week_feature[:, -1]
        year, month = last_year_month[:4], last_year_month[4:]
        next_ = datetime(int(year), int(month)+1, int(1)) if int(month)+1<13 else month
        next_month = next_.strftime('%Y%m')

        for i in range(len(date)):
            if date[i][:6] == next_month:
                pred_data = new_week_feature[:i]
                future_true_data = new_week_feature[i:]
                break

        train_data, train_label = pred_data[:, :-2], pred_data[:, -2]
        other_data = future_true_data[:, -2]
        date_ = np.hstack((future_true_data[:, -1], pred_date_feature[:, -1]))
        index = pd.to_datetime(np.asarray(date_), format=self.date_format)
        future_time = index
        index_history = pd.to_datetime(pred_data[:, -1], format=self.date_format)
        index_history = [x.strftime('%F') for x in index_history]


        x_train, _, y_train, _ = train_test_split(train_data, train_label, test_size=0)

        model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=30,
                                      learning_rate=0.01, n_estimators=43, max_depth=5,
                                      metric='rmse', bagging_fraction=0.8, feature_fraction=0.8)
        params_test = {'max_depth': range(3, 15, 1), 'num_leaves': range(10, 100, 10),
                       'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]}
        gbm = GridSearchCV(estimator=model_lgb, param_grid=params_test,
                           scoring='neg_mean_squared_error', cv=3, verbose=0, n_jobs=4)
        gbm.fit(x_train, y_train)


        pred_future = gbm.predict(pred_date_feature[:, :-1])

        other_data = future_true_data[:, -2]

        pred_history = gbm.predict(train_data)
        mre = np.average([np.abs((b - a)) / a for a, b in
                          zip(train_label.astype(np.float), pred_history.astype(np.float))])

        pred_all = np.hstack((other_data, pred_future))
        pred_all = [np.power(10, x) for x in pred_all]
        time_series = pd.Series(data=pred_history, index=index_history)
        predict_time_series = pd.Series(data=pred_all, index=future_time)
        pred_month = predict_time_series.resample('M').sum()

        month_index = pred_month.index
        month_index = [x.strftime('%Y%m') for x in month_index]
        month_data = pred_month.values

        str_ = ''
        for data, date in zip(month_data, list(month_index)):
            str_ += (str(date) + '_' + str(data) + ';')

        # mre = np.average([np.abs((b - a)) / a for a, b in zip(np.array(pred_data[:, -1]).astype(np.float), np.array(pred_history).astype(np.float))])
        mre = round(mre, 6)
        dict = {}
        dict['method'] = 'lightgbm'
        dict['pred_data'] = str_[:-1]
        dict["mre"] = mre
        dict['status'] = 'succeed'
        # accuracy = json.dumps(dict)

        return dict

# lstm model
from bilstm import LSTM
import torch
import torch.nn as nn
class Lstm_model():

    def __init__(self, step_back=3, epoch=100, hidden_size=5, num_layer=1, lr=0.02, bidirectional=False):
        '''
        :param
        :param
        :param
        '''
        self.step_back = step_back
        self.epoch = epoch
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lr = lr
        self.bidirectional = bidirectional

    def train_lstm(self, x_train, y_train):

        logger.info('training for lstm ==>>')
        lstm = LSTM(input_size=self.step_back, hidden_size=self.hidden_size, num_layer=self.num_layer, bidirectional=self.bidirectional)
        # 参数寻优，计算损失函数
        optimizer = torch.optim.Adam(lstm.parameters(), lr=self.lr)
        loss_func = nn.MSELoss()
        loss_list = []
        for i in range(self.epoch):
            out = lstm(x_train)
            loss = loss_func(out, y_train)
            loss_list.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # torch.save(lstm.state_dict(), 'x.pt')
        # torch.save(lstm, 'x.pt')
        # plt.plot(loss_list)
        # plt.show()
        #logger.info('每步的损失为{}'.format(loss_list))
        return lstm

    def test_lstm(self, x_train, y_train, x_test, y_test):

        lstm = self.train_lstm(x_train, y_train)
        predict = lstm(x_test)

        return predict, y_test

    # create pred_data
    def for_predata(self, last_data, last_label):

        last_data = torch.cat((last_data, last_label), dim=1)
        last_data = last_data[0][-self.step_back:].unsqueeze(0).unsqueeze(0)

        return last_data

    def predict_all(self, all_data, all_label, scaler, pred_date_num=3):

        last_data = all_data[-1, :, :]
        predict = all_label[-1, :, :]
        logger.info('最后一个特征为{}标签为{}'.format(last_data, predict))
        lstm = self.train_lstm(all_data, all_label)
        predata = predict
        for i in tqdm(range(pred_date_num)):
            time.sleep(0.1)
            predata_ = self.for_predata(last_data, predict)
            predict = lstm(predata_).squeeze(1)
            logger.info('预测的第{}个特征为{}，预测数据为{}'.format(i+1, predata_, predict))
            predata = torch.cat((predata, predict), dim=0)
            last_data = predata_.squeeze(0)
        pred_all_ = predata[1:, :]
        # 反归一化
        pred_history = lstm(all_data).squeeze(1)
        pred_history = scaler.inverse_transform(pred_history.detach().numpy())
        pred_all_ = scaler.inverse_transform(pred_all_.detach().numpy())

        step_back_data = all_data[0, :, :].detach().numpy()
        step_back_data = scaler.inverse_transform(step_back_data)
        step_back_data = step_back_data.reshape(-1, 1)
        pred_all = np.vstack((step_back_data, pred_history, pred_all_)).reshape(-1)

        return pred_all

# TCN
from common.tcn import TCN
import torch.nn.functional as function
class TCN_model():

    def __init__(self, step_back=3, lr=0.02, dropout=0.5, kernel_size=2, channel_sizes=7,
                 n_classes=1, input_channels=1, epoch=200):
        '''
        :param
        :param
        :param
        '''
        self.step_back = step_back
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.channel_sizes = [10]*channel_sizes
        self.kernel_size = kernel_size
        self.lr = lr
        self.dropout = dropout
        self.epoch = epoch


    def TCN_train(self, x_train, y_train):

        logger.info('training for TCN ==>>')
        tcn = TCN(self.input_channels, self.n_classes, self.channel_sizes, kernel_size=self.kernel_size, dropout=self.dropout)
        optimizer = torch.optim.Adam(tcn.parameters(), lr=self.lr)
        #optimizer = getattr(torch.optim, self.optim)(model.parameters(), lr=lr)
        loss_func = nn.MSELoss()
        loss_list = []

        for i in tqdm(range(self.epoch)):
            # model.train()
            # optimizer.zero_grad()
            output = tcn(x_train)
            loss = loss_func(output, y_train)
            loss_list.append(loss.detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        plt.plot(loss_list)
        plt.show()

        return tcn

    def TCN_test(self, x_train, y_train, x_test, y_test, scaler):

        y_train = y_train.squeeze(1)
        y_test = y_test.squeeze(1)
        tcn = self.TCN_train(x_train, y_train)
        predict = tcn(x_test)

        y_test = scaler.inverse_transform(y_test.detach().numpy())
        predict = scaler.inverse_transform(predict.detach().numpy())

        return predict, y_test

    # create pred_data
    def for_predata(self, last_data, last_label):
        last_data = torch.cat((last_data, last_label), dim=1)
        last_data = last_data[0][-self.step_back:].unsqueeze(0).unsqueeze(0)

        return last_data

    def predict_all(self, all_data, all_label, scaler, pred_date_num=3):
        last_data = all_data[-1, :, :]
        predict = all_label[-1, :, :]
        logger.info('最后一个特征为{}标签为{}'.format(last_data, predict))
        all_label = all_label.squeeze(1)
        Tcn = self.TCN_train(all_data, all_label)
        predata = predict
        for i in range(pred_date_num):
            predata_ = self.for_predata(last_data, predict)# ([1,3], [1,1])
            predict = Tcn(predata_)
            logger.info('预测的第{}个特征为{}，预测数据为{}'.format(i + 1, predata_, predict))
            predata = torch.cat((predata, predict), dim=0)# [2,1]
            last_data = predata_.squeeze(0)
        pred_all_ = predata[1:, :]
        # 反归一化
        pred_history = Tcn(all_data)
        pred_history = scaler.inverse_transform(pred_history.detach().numpy())
        pred_all_ = scaler.inverse_transform(pred_all_.detach().numpy())

        step_back_data = all_data[0, :, :].detach().numpy()
        step_back_data = scaler.inverse_transform(step_back_data)
        step_back_data = step_back_data.reshape(-1, 1)
        pred_all = np.vstack((step_back_data, pred_history, pred_all_)).reshape(-1)

        return pred_all

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

# 模型可解释性
class Explain():
    # 预测精度的解释性
    def __init__(self, num_preds=3, if_pred_all=True, method_str='unknown-method', date_format='%Y%m%d', Do_featuredata=False,
                 if_log10=False):
        '''
        :param
        :param
        :param
        '''
        self.num_preds = num_preds
        self.if_pred_all = if_pred_all
        self.method_str = method_str
        self.date_format = date_format
        self.Do_featuredata = Do_featuredata
        self.if_log10 = if_log10
        self.freq = self.format_type(date_format=date_format)

    def format_type(self, date_format='%Y%m%d'):
        if date_format == '%Y':
            freq = 'AS'
        elif date_format == '%Y%m':
            freq = 'MS'
        else:
            freq ='D'
        return freq

    def accuracy(self, original_series, predict):
        '''
        :param original_series: 原始数据序列
        :param pred: 预测数据序列
        :return: r2、mae等指标的json字符串
        '''
        if self.if_pred_all:
            pred = predict[:original_series.size]
            original_series = original_series.values
        else:
            pred = predict
            original_series = original_series.values[-self.num_preds:]

        r2 = r2_score(original_series, pred)  # r2
        mse = mean_squared_error(original_series, pred)
        rmse = np.sqrt(mse)  # rmse
        mae = mean_absolute_error(original_series, pred)  # mae
        # 平均相对误差计算:mean relative error/MRE
        mre = np.average([np.abs((b - a)) / a for a, b in zip(original_series, pred)])
        mre = round(mre, 6)
        dict = {}
        dict["r2"] = r2
        dict["rmse"] = rmse
        dict["mae"] = mae
        dict["mre"] = mre
        accuracy = json.dumps(dict)

        return accuracy, mre

    def predict_format(self, original_series, predict, remaining_data, future_time=None):
        # 封装预测结果
        _, mre = self.accuracy(original_series, predict)
        if self.if_log10:
            predict = [np.power(10, x) for x in predict]
            original_series = original_series.apply(lambda x: np.power(10, x))

        str_history, str_pred = self.to_str(original_series, predict, remaining_data=remaining_data, future_time=future_time)
        dict_pred = {}
        dict_pred["method"] = self.method_str
        dict_pred['pred_data'] = str_pred
        # dict_pred['history_data'] = str_history
        dict_pred['mre'] = mre
        dict_pred['info'] = 'successed'

        return dict_pred

    # 封装预测结果为字符串--格式为‘201401_1000000;201402_2000122’
    def to_str(self, time_series, predict, remaining_data, future_time=None):

        if self.if_pred_all:
            data_pred = predict[time_series.size:]
            data_history = time_series.values
            if self.Do_featuredata:
                index_all = pd.to_datetime(future_time, format=self.date_format)#.map(lambda x: x.strftime(self.date_format))
                index_all = pd.to_datetime(index_all).map(lambda x: x.strftime(self.date_format))

            else:
                index_all = pd.date_range(list(time_series.index)[0], periods=(time_series.size + self.num_preds),
                                          freq=self.freq)
                index_all = pd.to_datetime(index_all).map(lambda x: x.strftime(self.date_format))
        else:
            data_pred = predict
            data_history = time_series.values[:-self.num_preds]
            index_all = pd.date_range(list(time_series.index)[0], periods=(time_series.size),
                                      freq=self.freq)
            index_all = pd.to_datetime(index_all).map(lambda x: x.strftime(self.date_format))

        index_history = index_all[:-self.num_preds]
        index_pred = index_all[-self.num_preds:]
        str_history = self.to_str_1(data_history, index_history)
        logger.info('Compensation for predicted values')
        if remaining_data != np.array(data_pred).sum():
            data_pred += np.divide(np.array(data_pred), np.array(data_pred).sum()) * remaining_data
        str_pred = self.to_str_1(data_pred, index_pred)
        return str_history, str_pred

    # 封装预测结果为字符串
    def to_str_1(self, data_list, date_list):
        str_ = ''
        for data, date in zip(data_list, list(date_list)):
            str_ += (str(date) + '_' + str(data) + ';')
        return str_[:-1]

# 画图类
class plot_table():
    def __init__(self, num_preds=10, if_pred_all=True, date_format='%Y%m', if_log=False):
        '''
        :param num_preds：预测时间长度
        :param if_pred_all：是否所有的数据作为训练集
        :param
        '''
        self.num_preds = num_preds
        self.if_pred_all = if_pred_all
        self.freq = self.format_type(date_format=date_format)
        self.if_log = if_log

    def format_type(self, date_format='%Y%m%d'):
        if date_format == '%Y':
            freq = 'Y'
        elif date_format == '%Y%m':
            freq = 'M'
        else:
            freq = 'D'

        return freq

    def plot(self, original_series, predict):


        if self.if_log:
            original_series = original_series.apply(lambda x: np.power(10, x))
            predict = [np.power(10, x) for x in predict]

        if self.if_pred_all:
            pred = predict[:original_series.size]
            original_series_index = original_series.index
            original_series = original_series.values
        else:
            pred = predict
            original_series_index = original_series.index[-self.num_preds:]
            original_series = original_series.values[-self.num_preds:]
        plt.plot(original_series_index, pred, label="pred", color='b')
        plt.plot(original_series_index, original_series, label="orag", color='r')
        plt.legend(loc=0)
        plt.tick_params(axis='x', labelsize=8)
        plt.xticks(rotation=-90)
        plt.title("Comparison of raw data and forecast data")
        plt.subplots_adjust(bottom=0.2)
        plt.show()


if __name__ == '__main__':

    feature = lightgbm_holidays(ColumnDate, ColumnData, date_format=date_format, if_log=True, year=2020, month=4,
                                sql_data=sql_day_data_11)
    new_week_feature, pred_feature, last_year_month = feature.split_features(8, csv_path)

    lightgbm = Lightgbm_model(ratio=ratio, date_format=date_format)
    accuracy = lightgbm.predict_all(new_week_feature, pred_feature, last_year_month)
