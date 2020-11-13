# -*- coding:utf-8 -*-
# @Time: 2020/10/22 10:14
# @File: Data_Preprocess.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
#from __future__ import absolute_import
from config import *
from holiday import *
from log import logger, init_logger
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
init_logger()#log_path)
logger.info('==>start data preprocessing')

import cx_Oracle
# 数据读取和预处理类
class Do_Properties():
    def __init__(self, properties_path):# host=None, port=None, sid=None, username=None, password=None):
        self.prop = Properties(properties_path)
        self.dict_1 = self.prop.getProperties()
        self.oracle = self.dict_1["oracle"]
        self.host = self.oracle["host"]
        self.port = self.oracle["port"]
        self.sid = self.oracle["sid"]
        self.username = self.oracle["username"]
        self.password = self.oracle["password"]
        self.sql = self.dict_1["sql"]
        self.limit_num = int(self.dict_1['limit_num'])

    def to_series(self, df, index_name, column_name):  # 数据处理为Series
        # tax_data.columns = ["YYYYMM", "SJ"]
        series_all = pd.Series(df[column_name].values, index=df[index_name])
        return series_all

    def oracle_query(self, sql_data):  # 查询ORACLE数据
        sql = self.sql[sql_data]
        try:
            dsn = cx_Oracle.makedsn(self.host, self.port, self.sid)
            conn = cx_Oracle.connect(self.username, self.password, dsn)  # 用户名、密码
            results = pd.read_sql(sql, conn)
        except (UnboundLocalError, cx_Oracle.DatabaseError):
            print("数据库的用户名、密码或者端口号错误")
        except pd.io.sql.DatabaseError:
            print("sql语句错误")
        else:
            conn.close()
            return results

    # 创建表
    def create_table(self, sql):
        '''
        :param sql: sql语句
        :return: 无
        '''
        dsn = cx_Oracle.makedsn(self.host, self.port, self.sid)
        conn = cx_Oracle.connect(self.username, self.password, dsn)  # "nx_czsb", "oracle",
        cr = conn.cursor()
        cr.execute(sql)
        cr.execute("commit")
        conn.commit()
        conn.close()

    # 插入数据
    def excute_sql(self, sql, params):
        '''
        :param sql: sql语句
        :param params: 需要存储的数据
        :return:
        '''
        dsn = cx_Oracle.makedsn(self.host, self.port, self.sid)
        conn = cx_Oracle.connect(self.username, self.password, dsn)
        cr = conn.cursor()
        cr.executemany(sql, params)
        cr.execute("commit")
        conn.commit()
        conn.close()

# 数据处理类
class Do_TimeData():
    def __init__(self, ColumnDate, ColumnData, format='%Y%m', year=2020, month=4, sql_data=None):
        '''
        input:
        :param ColumnDate: 日期列
        :param ColumnData: 数据列
        :param format: 日期格式
        :param year: 预测年份
        :param month: 已完成月份
        :param sql_data: SQL语句
        '''
        self.format = format
        self.ColumnDate = ColumnDate
        self.ColumnData = ColumnData
        self.sql_data = sql_data
        self.year = year
        self.month = month

    def read_time_data(self, csv_path=None, limit_num=24):
        '''
        input:
        :param csv_path: 如果不为None，则读地址内容，否则数据库
        :param limit_num: 数据最少个数
        :return:
        time_series：时间序列数据
        '''
        # read csv or oracle
        global date_index
        if csv_path is None:
            df = Do_Properties(properties_path).oracle_query(self.sql_data)
            logger.info('read data from oracle')
        else:
            df = pd.read_csv(csv_path)
            logger.info('read data from %s' % csv_path)
        logger.info('将DataFrame的数据格式变成Series格式')

        last_year_month = str(self.year) + ('0' if self.month < 10 else '') + str(self.month)


        if len(df) >= limit_num and last_year_month not in df[self.ColumnDate].values and self.format == '%Y%m':
            logger.info('数据库缺少到预测年份已完成月份{}前的历史数据'.format(last_year_month))
            return {'status': 'failed', 'results': '数据库缺少到已完成月份{}前的历史数据'.format(last_year_month)}

        logger.info('last year_month is {}'.format(last_year_month))
        loc = df[df[self.ColumnDate] == last_year_month].index[0]
        df = df.iloc[:loc + 1]

        if df[self.ColumnDate].size <= limit_num:
            logger.info("数据少")
            last_date = last_year_month
            if self.format == '%Y%m%d':
                year, month, date = last_date[:4], last_date[4:6], last_date[6:8]
                last_date = datetime(int(year), int(month), int(date))
                pred_last_date = last_date + timedelta(days=(12 - int(self.month)))
                date_index = pd.date_range(last_date, pred_last_date)
                date_index = [x.strftime(self.format) for x in date_index]
            elif self.format == '%Y%m':
                year, month = last_date[:4], last_date[4:6]
                last_date = datetime(int(year), int(month), 1)
                pred_last_date = last_date + timedelta(days=(12 - int(self.month)))
                date_index = pd.date_range(start=pred_last_date, periods=(12 - int(self.month)), freq='MS')
                date_index = [x.strftime(self.format) for x in date_index]
            elif self.format == '%Y':  # 未测试
                last_date = df[self.ColumnDate].iloc[-1]
            return {'status': 'successed', 'date_index': date_index}

        date_index = pd.to_datetime(df[self.ColumnDate], format=self.format)
        time_series = pd.Series(data=df[self.ColumnData].values, index=date_index)

        return time_series

    def data_for_train(self, time_series, if_log10=False, zero_ratio=3):
        '''
        input：
        :param time_series: 时间序列数据
        :param if_log10: 是否log10变换
        :param zero_ratio: 当不正常数据大于三分之一时不预测
        :return:
        timeseries:时间学列数据
        '''
        Positive_number = time_series[time_series <= 0].size
        if Positive_number >= int(time_series.size / zero_ratio):
            logger.info("数据异常值太多，不适合预测")
            return {'status': 'failed', 'results': '数据异常值太多，不适合预测'}
        # 填补空值和处理小于0的值
        time_series = time_series.fillna(method='bfill')
        time_series = time_series.fillna(method='ffill')
        time_series[time_series <= 0] = 0.1
        # 如果不连续，填补缺值
        if self.format == '%Y':
            freq = 'As'
        elif self.format == '%Y%m':
            freq = 'MS'
        else:
            freq = 'D'
        if len(pd.date_range(time_series.index[0], end=time_series.index[-1], freq=freq)) > time_series.size:
            index_all = pd.date_range(time_series.index[0], end=time_series.index[-1], freq=freq)
            index_miss = list(set(index_all).difference(set(time_series.index)))
            time_series = time_series.append(
                pd.Series(data=[0.1 for x in range(len(index_miss))], index=pd.to_datetime(index_miss))
            ).sort_index()

        if if_log10:
            time_series = time_series.apply(lambda x: np.log10(x))
        else:
            time_series = time_series
        return time_series

# data for lightgbm
class Do_FeatureData():

    def __init__(self, ColumnDate, ColumnData, date_format='%Y%m', if_log=True, year=2020, month=4, sql_data=None):
        '''
        input:
        :param ColumnDate: 日期列
        :param ColumnData: 数据列
        :param date_format: 日期格式
        :param if_log: 是否log10变换
        :param year: 预测年份
        :param month: 已完成月份
        :param sql_data: SQL语句
        '''
        self.date_format = date_format
        self.ColumnDate = ColumnDate
        self.ColumnData = ColumnData
        self.if_log = if_log
        self.sql_data = sql_data
        self.year = year
        self.month = month

    def split_features(self, pred_date_num, csv_path=None, limit_num=24):
        '''
        input:
        :param pred_date_num: 要预测的时间长度
        :param csv_path: 如果不为None，则读地址内容，否则数据库
        :param limit_num: 数据最少个数
        :return:
        new_week_feature：拆分的时间特征
        pred_date_feature：要预测的时间特征
        '''
        # read csv or oracle
        global date_index
        if csv_path is None:
            df = Do_Properties(properties_path).oracle_query(self.sql_data)
            logger.info('read data from oracle')
        else:
            df = pd.read_csv(csv_path)
            logger.info('read data from %s' % csv_path)

        last_year_month = str(self.year) + ('0' if self.month < 10 else '') + str(self.month)
        logger.info('last year_month is {}'.format(last_year_month))
        if len(df) >= limit_num and last_year_month not in df[self.ColumnDate].values and self.date_format == '%Y%m':
            logger.info('数据库缺少到预测年份已完成月份{}前的历史数据'.format(last_year_month))
            return {'status': 'failed', 'results': '数据库缺少到已完成月份{}前的历史数据'.format(last_year_month)}, None

        loc = df[df[self.ColumnDate] == last_year_month].index[0]
        df = df.iloc[:loc + 1]

        if df[self.ColumnDate].size <= limit_num:
            logger.info("数据少")
            last_date = last_year_month
            if self.date_format == '%Y%m%d':
                year, month, date = last_date[:4], last_date[4:6], last_date[6:8]
                last_date = datetime(int(year), int(month), int(date))
                pred_last_date = last_date + timedelta(days=pred_date_num)
                date_index = pd.date_range(last_date, pred_last_date)
                date_index = [x.strftime(self.date_format) for x in date_index]
            elif self.date_format == '%Y%m':
                year, month = last_date[:4], last_date[4:6]
                last_date = datetime(int(year), int(month), 1)
                pred_last_date = last_date + timedelta(days=pred_date_num)
                date_index = pd.date_range(start=pred_last_date, periods=pred_date_num, freq='MS')
                date_index = [x.strftime(self.date_format) for x in date_index]
            elif self.date_format == '%Y':  # 未测试
                last_date = df[self.ColumnDate].iloc[-1]
            return {'status': 'successed', 'date_index': date_index}, None

        # df[self.ColumnData].loc[df[self.ColumnData] < 0] = df[self.ColumnData].loc[df[self.ColumnData] > 0].values.mean()
        date_index = pd.to_datetime(df[self.ColumnDate], format=self.date_format)
        new_week_feature = []
        for i in date_index:
            if self.date_format == '%Y%m':
                logger.info('Historical data: The characteristics of the month')
                # 年月切分
                year, month = str(i)[:7].split('-')
                # 第几季度
                if int(month) in [1, 2, 3]:
                    month_num = 1
                elif int(month) in [4, 5, 6]:
                    month_num = 2
                elif int(month) in [7, 8, 9]:
                    month_num = 3
                else:
                    month_num = 4
                if self.if_log:
                    new_feature = [np.log(int(year)), int(month), month_num]
            elif self.date_format == '%Y%m%d':
                logger.info('Historical data: The characteristics of the day')
                str_dates = str(i)[:10]
                # 判断周几
                week_num = datetime.strptime(str_dates, "%Y-%m-%d").weekday() + 1
                # 判断第几周
                week_Nums = int(datetime.strptime(str_dates, '%Y-%m-%d').strftime('%W'))
                # 日期切分： 年-月-日
                year, month, date = str_dates.split('-')
                if int(month) in [1, 2, 3]:
                    month_num = 1
                elif int(month) in [4, 5, 6]:
                    month_num = 2
                elif int(month) in [7, 8, 9]:
                    month_num = 3
                else:
                    month_num = 4
                if self.if_log:
                    new_feature = [np.log(int(year)), int(month), int(date), month_num, week_num, week_Nums]
            elif self.date_format == '%Y':#未测试
                new_week_feature = np.hstack((list(df[self.ColumnDate].values), list(df[self.ColumnData].values)))
            else:
                logger.info("请输入正确的日期格式")
                return {'status': 'failed', 'results': '日期格式错误'}
            new_week_feature.append(new_feature)
        if self.if_log:
            label = df[[self.ColumnData]].apply(lambda x: np.log10(x)).values
        date_index = date_index.apply(lambda x: x.strftime(self.date_format))
        new_week_feature = np.hstack((new_week_feature, label, date_index.values.reshape(-1, 1)))

        # 预测的特征
        logger.info('Structural prediction characteristics ==>>')
        last_date = str(df[self.ColumnDate].iloc[-1])
        pred_date_feature = []
        index_date = []
        if self.date_format == '%Y%m%d':
            logger.info('Predict data: The characteristics of the day')
            year, month, date = last_date[:4], last_date[4:6], last_date[6:8]
            last_date = datetime(int(year), int(month), int(date))
            pred_last_date = last_date + timedelta(days=pred_date_num)
            date_index = pd.date_range(last_date, pred_last_date).values
            for i in date_index:
                str_dates = str(i)[:10]
                index_date.append(str_dates)
                # 判断周几
                week_num = datetime.strptime(str_dates, "%Y-%m-%d").weekday() + 1
                # 判断第几周
                week_Nums = int(datetime.strptime(str_dates, '%Y-%m-%d').strftime('%W'))
                # 日期切分： 年-月-日
                year, month, date = str_dates.split('-')
                if int(month) in [1, 2, 3]:
                    month_num = 1
                elif int(month) in [4, 5, 6]:
                    month_num = 2
                elif int(month) in [7, 8, 9]:
                    month_num = 3
                else:
                    month_num = 4
                if self.if_log:
                    new_feature = [np.log(int(year)), int(month), int(date), month_num, week_num, week_Nums]
                pred_date_feature.append(new_feature)
            date_index = [[str(x)[:10].replace('-', '')] for x in date_index]
            pred_date_feature = np.hstack((pred_date_feature[1:], date_index[1:]))
        elif self.date_format == '%Y%m':
            logger.info('Predict data: The characteristics of the month')
            year, month = last_date[:4], last_date[4:6]
            last_date = datetime(int(year), int(month), 1)
            pred_last_date = last_date + timedelta(days=pred_date_num)
            date_index = pd.date_range(start=pred_last_date, periods=pred_date_num, freq='MS')
            for i in date_index:
                str_dates = str(i)[:7]
                year, month = str_dates.split('-')
                if int(month) in [1, 2, 3]:
                    month_num = 1
                elif int(month) in [4, 5, 6]:
                    month_num = 2
                elif int(month) in [7, 8, 9]:
                    month_num = 3
                else:
                    month_num = 4
                if self.if_log:
                    new_feature = [np.log(int(year)), int(month), month_num]
                pred_date_feature.append(new_feature)
            date_index = [[str(x)[:7].replace('-', '')] for x in date_index.values]
            pred_date_feature = np.hstack((pred_date_feature, date_index))
        elif self.date_format == '%Y':  # 未测试
            last_date = df[self.ColumnDate].iloc[-1]
            for i in range(1, pred_date_num+1):
                pred_date_feature.append(int(last_date)+i)

        return new_week_feature, pred_date_feature


# data for lstm input
import torch
from sklearn.preprocessing import MinMaxScaler
class Do_for_lstm():

    def __init__(self, ColumnDate, ColumnData, date_format, zero_ratio=3, properties_path=None, year=2020, month=4, sql_data=None):
        '''
        input:
        :param ColumnDate: 日期列
        :param ColumnData: 数据列
        :param date_format: 日期格式
        :param zero_ratio: 异常数据不能大于1/zero_ratio
        :param properties_path: properties_path地址
        :param year: 预测年份
        :param month: 已完成月份
        :param sql_data: SQL语句
        '''
        self.properties_path = properties_path
        self.sql_data = sql_data
        self.ColumnDate = ColumnDate
        self.ColumnData = ColumnData
        self.date_format = date_format
        self.zero_ratio = zero_ratio
        self.year = year
        self.month = month

    def read_time_data(self, csv_path=None, limit_num=24):
        '''
        input:
        :param csv_path: 地址
        :param if_log10: 是否log10变换
        :param limit_num: 最少时间序列数
        output：
        time_series：时间序列
        '''
        # read csv or oracle
        global date_index
        if csv_path is None:

            df = Do_Properties(self.properties_path).oracle_query(self.sql_data)
            logger.info('read data from oracle')
        else:
            df = pd.read_csv(csv_path)
            logger.info('read data from %s' % csv_path)
        last_year_month = str(self.year) + ('0' if self.month < 10 else '') + str(self.month)
        logger.info('last year_month is {}'.format(last_year_month))
        if len(df) >= limit_num and last_year_month not in df[self.ColumnDate].values and self.date_format == '%Y%m':
            logger.info('数据库缺少到预测年份已完成月份{}前的历史数据'.format(last_year_month))
            return {'status': 'failed', 'results': '数据库缺少到已完成月份{}前的部分历史数据'.format(last_year_month)}
        loc = df[df[self.ColumnDate] == last_year_month].index[0]
        df = df.iloc[:loc + 1]
        if df[self.ColumnDate].size <= limit_num:
            logger.info("数据少")
            last_date = last_year_month
            if self.date_format == '%Y%m%d':
                year, month, date = last_date[:4], last_date[4:6], last_date[6:8]
                last_date = datetime(int(year), int(month), int(date))
                pred_last_date = last_date + timedelta(days=(12 - int(self.month)))
                date_index = pd.date_range(last_date, pred_last_date)
                date_index = [x.strftime(self.date_format) for x in date_index]
            elif self.date_format == '%Y%m':
                year, month = last_date[:4], last_date[4:6]
                last_date = datetime(int(year), int(month), 1)
                pred_last_date = last_date + timedelta(days=(12 - int(self.month)))
                date_index = pd.date_range(start=pred_last_date, periods=(12 - int(self.month)), freq='MS')
                date_index = [x.strftime(self.date_format) for x in date_index]
            elif self.date_format == '%Y':  # 未测试
                last_date = df[self.ColumnDate].iloc[-1]
            return {'status': 'successed', 'date_index': date_index}

        Positive_number = len(df[df[self.ColumnData] <= 0])
        if Positive_number >= len(df) / self.zero_ratio:
            logger.info("数据异常值太多，不适合预测")
            return {'status': 'failed', 'results': '数据异常值太多，不适合预测'}

        # df[self.ColumnData].loc[df[self.ColumnData] < 0] = df[self.ColumnData].loc[df[self.ColumnData] > 0].values.mean()
        date_index = pd.to_datetime(df[self.ColumnDate], format=self.date_format)
        time_series = pd.Series(data=df[self.ColumnData].values, index=date_index)

        # 填补空值和处理小于0的值
        time_series = time_series.fillna(method='bfill')
        time_series = time_series.fillna(method='ffill')

        # 如果不连续，填补缺值
        if self.date_format == '%Y':
            freq = 'As'
        elif self.date_format == '%Y%m':
            freq = 'MS'
        else:
            freq = 'D'
        if len(pd.date_range(time_series.index[0], end=time_series.index[-1], freq=freq)) > time_series.size:
            index_all = pd.date_range(time_series.index[0], end=time_series.index[-1], freq=freq)
            index_miss = list(set(index_all).difference(set(time_series.index)))
            time_series = time_series.append(
                pd.Series(data=[0.1 for x in range(len(index_miss))], index=pd.to_datetime(index_miss))
            ).sort_index()

        return time_series

    def data_for_train(self, csv_path=None, step_back=3, test_ratio=0.2):
        '''
        input:
        :param csv_path: 地址
        :param step_back: 回溯步数
        :param test_ratio: 测试集比率
        output：
        x_train：训练集特征
        y_train：训练集标签
        x_test：测试特征
        y_test：测试标签
        dataX：所有特征
        dataY：所有标签
        scaler：归一化模块
        '''
        time_series = self.read_time_data(csv_path=csv_path)
        # 将数据归一化
        scaler_data = time_series.values.reshape((time_series.size, 1))
        # # 公式
        # X = scaler_data
        # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        # max = 0.9
        # min = 0.1
        # X_scaled = X_std * (max - min) + min
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        scaler = scaler.fit(scaler_data)
        scaler_data = scaler.transform(scaler_data)
        # 划分训练集和测试集
        data_x = []
        data_y = []
        for i in range(len(scaler_data) - step_back):
            data_x.append(scaler_data[i:i + step_back])
            data_y.append(scaler_data[i + step_back])

        dataX, dataY = np.asarray(data_x), np.asarray(data_y)
        train_size = int(len(dataX) * (1 - test_ratio))
        x_train, y_train, x_test, y_test = dataX[:train_size], dataY[:train_size], dataX[train_size:], dataY[train_size:]

        dataX = dataX.reshape(-1, 1, step_back)
        dataY = dataY.reshape(-1, 1, 1)
        x_train = x_train.reshape(-1, 1, step_back)
        y_train = y_train.reshape(-1, 1, 1)
        x_test = x_test.reshape(-1, 1, step_back)
        y_test = y_test.reshape(-1, 1, 1)

        dataX = torch.tensor(dataX, dtype=torch.float32)
        dataY = torch.tensor(dataY, dtype=torch.float32)
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        return x_train, y_train, x_test, y_test, dataX, dataY, scaler

    def time_for_pred(self, csv_path=None, pred_date_num=3, limit_num=24):
        '''
        input:
        :param csv_path: 地址
        :param pred_date_num: 预测时间长度
        :param limit_num: 最少时间数据个数
        output：
        date_index：预测时间特征
        '''
        # read csv or oracle
        global date_index
        if csv_path is None:

            df = Do_Properties(self.properties_path).oracle_query(self.sql_data)
            logger.info('read data from oracle')
        else:
            df = pd.read_csv(csv_path)
            logger.info('read data from %s' % csv_path)

        last_year_month = str(self.year) + ('0' if self.month < 10 else '') + str(self.month)
        logger.info('last year_month is {}'.format(last_year_month))
        if len(df) >= limit_num and last_year_month not in df[self.ColumnDate].values and self.date_format == '%Y%m':
            logger.info('数据库缺少到预测年份已完成月份{}前的历史数据'.format(last_year_month))
            return {'status': 'failed', 'results': '数据库缺少到已完成月份{}前的部分历史数据'.format(last_year_month)}
        loc = df[df[self.ColumnDate] == last_year_month].index[0]
        df = df.iloc[:loc + 1]
        if df[self.ColumnDate].size <= limit_num:
            logger.info("数据少")
            last_date = last_year_month
            if self.date_format == '%Y%m%d':
                year, month, date = last_date[:4], last_date[4:6], last_date[6:8]
                last_date = datetime(int(year), int(month), int(date))
                pred_last_date = last_date + timedelta(days=(12 - int(self.month)))
                date_index = pd.date_range(last_date, pred_last_date)
                date_index = [x.strftime(self.date_format) for x in date_index]
            elif self.date_format == '%Y%m':
                year, month = last_date[:4], last_date[4:6]
                last_date = datetime(int(year), int(month), 1)
                pred_last_date = last_date + timedelta(days=(12 - int(self.month)))
                date_index = pd.date_range(start=pred_last_date, periods=(12 - int(self.month)), freq='MS')
                date_index = [x.strftime(self.date_format) for x in date_index]
            elif self.date_format == '%Y':  # 未测试
                last_date = df[self.ColumnDate].iloc[-1]
            return {'status': 'successed', 'date_index': date_index}

        Positive_number = len(df[df[self.ColumnData] <= 0])
        if Positive_number >= len(df) / self.zero_ratio:
            logger.info("数据异常值太多，不适合预测")
            return {'status': 'failed', 'results': '数据异常值太多，不适合预测'}

        history_time = df[self.ColumnDate].values
        history_time = [str(x) for x in history_time]
        last_date = str(df[self.ColumnDate].iloc[-1])
        if self.date_format == '%Y%m%d':
            year, month, date = last_date[:4], last_date[4:6], last_date[6:8]
            last_date = datetime(int(year), int(month), int(date))
            pred_last_date = last_date + timedelta(days=pred_date_num)
            date_index = pd.date_range(last_date, pred_last_date)
            date_index = [x.strftime(self.date_format) for x in date_index]
            date_index = np.hstack((history_time, date_index))

        elif self.date_format == '%Y%m':
            year, month = last_date[:4], last_date[4:6]
            last_date = datetime(int(year), int(month), 1)
            pred_last_date = last_date + timedelta(days=pred_date_num)
            date_index = pd.date_range(start=pred_last_date, periods=pred_date_num, freq='MS')
            date_index = [x.strftime(self.date_format) for x in date_index]
            date_index = np.hstack((history_time, date_index))

        elif self.date_format == '%Y':  # 未测试
            last_date = df[self.ColumnDate].iloc[-1]
            # for i in range(1, pred_date_num+1):
                # pred_date_feature.append(int(last_date)+i)

        return date_index


# 解析properties文件
class Properties(object):

    def __init__(self, filepathName):
        self.fileName = filepathName
        self.properties = {}

    def __getDict(self, strName, dictName, value):

        if (strName.find('.') > 0):
            k = strName.split('.')[0]
            dictName.setdefault(k, {})
            return self.__getDict(strName[len(k) + 1:], dictName[k], value)
        else:
            dictName[strName] = value
            return

    def getProperties(self):
        try:
            pro_file = open(self.fileName, 'Ur', encoding='utf-8')
            for line in pro_file.readlines():
                line = line.strip().replace('\n', '')
                if line.find("#") != -1:
                    line = line[0:line.find('#')]
                if line.find('=') > 0:
                    strs = line.split('=')
                    strs[1] = line[len(strs[0]) + 1:]
                    self.__getDict(strs[0].strip(), self.properties, strs[1].strip())
        except Exception:
            raise Exception
        else:
            pro_file.close()
        return self.properties


# data for lightgbm
class lightgbm_holidays():

    def __init__(self, ColumnDate, ColumnData, date_format='%Y%m', if_log=True, year=2020, month=4, sql_data=None):
        '''
        input:
        :param ColumnDate: 日期列
        :param ColumnData: 数据列
        :param date_format: 日期格式
        :param if_log: 是否log10变换
        :param year: 预测年份
        :param month: 已完成月份
        :param sql_data: SQL语句
        '''
        self.date_format = date_format
        self.ColumnDate = ColumnDate
        self.ColumnData = ColumnData
        self.if_log = if_log
        self.sql_data = sql_data
        self.year = year
        self.month = month

    def split_features(self, pred_date_num, csv_path=None, limit_num=500):
        '''
        input:
        :param pred_date_num: 要预测的时间长度
        :param csv_path: 如果不为None，则读地址内容，否则数据库
        :param limit_num: 数据最少个数
        :return:
        new_week_feature：拆分的时间特征
        pred_date_feature：要预测的时间特征
        '''
        # read csv or oracle
        global date_index, new_feature, label
        if csv_path is None:
            df = Do_Properties(properties_path).oracle_query(self.sql_data)
            logger.info('read data from oracle')
        else:
            df = pd.read_csv(csv_path)
            logger.info('read data from %s' % csv_path)

        last_year_month = str(self.year) + ('0' if self.month < 10 else '') + str(self.month)

        logger.info('last year_month is {}'.format(last_year_month))

        if df[self.ColumnDate].size <= limit_num:
            logger.info("数据少，返回年初预算总值和已完成数值差值除以预测月个数")

            date_index = pd.date_range(start=last_year_month+'01', periods=pred_date_num+1, freq='MS')

            date_index = [x.strftime('%Y%m') for x in date_index[1:]]

            return {'status': 'successed', 'date_index': date_index}, None, None

        df[self.ColumnData].loc[df[self.ColumnData] < 0] = df[self.ColumnData].loc[df[self.ColumnData] > 0].values.mean()
        date_index = pd.to_datetime(df[self.ColumnDate], format=self.date_format)
        new_week_feature = []
        for i in date_index:
            logger.info('Historical data: The characteristics of the day')
            str_dates = str(i)[:10]
            # 判断周几
            week_num = datetime.strptime(str_dates, "%Y-%m-%d").weekday() + 1
            # 判断第几周
            week_Nums = int(datetime.strptime(str_dates, '%Y-%m-%d').strftime('%W'))
            # 日期切分： 年-月-日
            year, month, date = str_dates.split('-')
            if int(month) in [1, 2, 3]:
                month_num = 1
            elif int(month) in [4, 5, 6]:
                month_num = 2
            elif int(month) in [7, 8, 9]:
                month_num = 3
            else:
                month_num = 4
            if self.if_log:
                new_feature = [np.log(int(year)), int(month), int(date), month_num, week_num, week_Nums]
            new_week_feature.append(new_feature)
        if self.if_log:
            label = df[[self.ColumnData]].apply(lambda x: np.log10(x)).values
        date_index = date_index.apply(lambda x: x.strftime(self.date_format))
        new_week_feature = np.hstack((new_week_feature, label, date_index.values.reshape(-1, 1)))

        # 预测的特征
        logger.info('Structural prediction characteristics ==>>')
        last_date = str(df[self.ColumnDate].iloc[-1])
        pred_date_feature = []
        index_date = []
        logger.info('Predict data: The characteristics of the day')
        year, month, date = last_date[:4], last_date[4:6], last_date[6:8]
        last_date = datetime(int(year), int(month), int(date))
        # pred_last_date = last_date + timedelta(days=pred_date_num)
        date_index_ = pd.date_range(last_date, str(self.year)+'1231')
        date_index_ = [x.strftime('%F') for x in date_index_]
        date_index = []
        work_days = all_holiday(self.year)
        work_days.extend(all_holiday(self.year+1))
        for x in date_index_:
            if x in work_days:
                date_index.extend([x])
        logger.info('all over work days ==> {}'.format(date_index[1:]))

        for i in date_index[1:]:
            str_dates = str(i)[:10]
            index_date.append(str_dates)
            # 判断周几
            week_num = datetime.strptime(str_dates, "%Y-%m-%d").weekday() + 1
            # 判断第几周
            week_Nums = int(datetime.strptime(str_dates, '%Y-%m-%d').strftime('%W'))
            # 日期切分： 年-月-日
            year, month, date = str_dates.split('-')
            if int(month) in [1, 2, 3]:
                month_num = 1
            elif int(month) in [4, 5, 6]:
                month_num = 2
            elif int(month) in [7, 8, 9]:
                month_num = 3
            else:
                month_num = 4
            if self.if_log:
                new_feature = [np.log(int(year)), int(month), int(date), month_num, week_num, week_Nums]
            pred_date_feature.append(new_feature)

        date_index = [x.replace('-', '') for x in date_index[1:]]

        pred_date_feature = np.hstack((pred_date_feature, np.asarray(date_index).reshape(-1, 1)))

        return new_week_feature, pred_date_feature, last_year_month


if __name__ == '__main__':
    feature = lightgbm_holidays(ColumnDate, ColumnData, date_format=date_format, if_log=True, year=2020, month=4,
                             sql_data=sql_day_data_11)
    new_week_feature, pred_feature, last_year_month = feature.split_features(8, csv_path)