# -*- coding:utf-8 -*-
# @Time: 2020/11/3 9:11
# @File: pred_model.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
from common.log import logger, init_logger
from common.Data_Preprocess import *
from common.models import *
from common.config import *
import argparse
init_logger()#log_path)
logger.info('==>start')




# 总体实现
class Treasury_cash():

    def __init__(self, foreMethod, incomeType, year, budgetNum, month, completedNum):
        '''
        :param
        :param
        :param
        '''
        self.algorithm = foreMethod
        self.type = incomeType
        self.budgetNum = budgetNum
        self.finish_month = month
        self.completedNum = completedNum
        self.year = year
        self.pred_date_num = self.for_pred_date_num()
        self.remaining_data = self.for_remaining_data()
        self.sql_data = self.sql_data(date_format)



    # 根据month返回预测长度date_format
    def for_pred_date_num(self):

        pred_date_num = 12 - self.finish_month
        logger.info('Prediction length is {}'.format(pred_date_num))
        return pred_date_num

    # for remaining data
    def for_remaining_data(self):

        remaining_data = self.budgetNum - self.completedNum
        logger.info('Remaining data is {}'.format(remaining_data))
        return remaining_data

    # 根据算法选择合适的算法
    def choose_algorithm(self):
        try:
            if self.algorithm == 'TCN':

                dict_pred = pred_by_TCN(csv_path, ColumnDate, ColumnData, date_format, self.pred_date_num,
                                        properties_path, self.sql_data,
                                        step_back, test_ratio, Do_featuredata, self.year, self.finish_month,
                                        self.remaining_data)

                return dict_pred
            elif self.algorithm == 'LSTM':

                dict_pred = pred_by_LSTM(csv_path, ColumnDate, ColumnData, date_format, properties_path, self.sql_data,
                                         self.pred_date_num,
                                         step_back, test_ratio, Do_featuredata, self.year, self.finish_month,
                                         self.remaining_data)

                return dict_pred
            elif self.algorithm == 'LightGBM':

                dict_pred = pred_by_LightGBM(csv_path, ColumnDate, ColumnData, date_format, self.pred_date_num, ratio,
                                             Do_featuredata, self.year, self.finish_month, self.remaining_data,
                                             sql_data=self.sql_data)

                return dict_pred
            elif self.algorithm == 'Prophet':

                dict_pred = pred_by_Prophet(csv_path, ColumnDate, ColumnData, date_format, self.pred_date_num,
                                            Do_featuredata, self.year, self.finish_month, self.remaining_data,
                                            sql_data=self.sql_data)

                return dict_pred
            elif self.algorithm == 'X11_Arima':

                dict_pred = pred_by_X11_Arima(csv_path, ColumnDate, ColumnData, date_format, self.pred_date_num,
                                              self.year, self.finish_month, self.remaining_data, sql_data=self.sql_data)

                return dict_pred
            elif self.algorithm == 'Arima':

                dict_pred = pred_by_Arima(csv_path, ColumnDate, ColumnData, date_format, self.pred_date_num, self.year,
                                          self.finish_month, self.remaining_data, sql_data=self.sql_data)

                return dict_pred
            else:
                return {"result": "fail"}
        except Exception as e:
            res = {'status': 'failed', 'results': str(e)}
            return json.dumps(res)


# Arima
def pred_by_Arima(csv_path, ColumnDate, ColumnData, date_format, pred_date_num, year, month, remaining_data, sql_data):

    Do_data = Do_TimeData(ColumnDate, ColumnData, format=date_format, year=year, month=month, sql_data=sql_data)
    time_series = Do_data.read_time_data(csv_path, limit_num)

    if isinstance(time_series, dict) and time_series['status'] == 'failed':
        return time_series['results']
    elif isinstance(time_series, dict) and time_series['status'] == 'successed':
        predict = [np.divide(remaining_data, pred_date_num)]*pred_date_num
        date_index = time_series['date_index']
        str_ = ''
        for data, date in zip(predict, list(date_index)):
            str_ += (str(date) + '_' + str(data) + ';')
        dict_pred = {}
        dict_pred['method'] = 'Arima'
        dict_pred['pred_data'] = str_[:-1]
        dict_pred['mre'] = 0.0001
        dict_pred['info'] = time_series['status']
        return dict_pred

    time_series = Do_data.data_for_train(time_series, if_log10=False, zero_ratio=3)
    logger.info('arima model')
    arima_pred = ARIMA_model(m=1, num_preds=pred_date_num, seasonal=True)
    predict = arima_pred.predict_all(time_series)

    explain = Explain(num_preds=pred_date_num, if_pred_all=True, method_str='arima', date_format=date_format,
                      if_log10=False)
    dict_pred = explain.predict_format(time_series, predict, remaining_data=remaining_data)

    if if_debug:
        plot = plot_table(num_preds=pred_date_num, date_format=date_format, if_log=False)
        plot.plot(time_series, predict)

    return dict_pred

# X11_Arima
def pred_by_X11_Arima(csv_path, ColumnDate, ColumnData, date_format, pred_date_num, year, month, remaining_data, sql_data):

    Do_data = Do_TimeData(ColumnDate, ColumnData, format=date_format, year=year, month=month, sql_data=sql_data)
    time_series = Do_data.read_time_data(csv_path, limit_num)

    if isinstance(time_series, dict) and time_series['status'] == 'failed':
        return time_series['results']
    elif isinstance(time_series, dict) and time_series['status'] == 'successed':
        predict = [np.divide(remaining_data, pred_date_num)] * pred_date_num
        date_index = time_series['date_index']
        str_ = ''
        for data, date in zip(predict, list(date_index)):
            str_ += (str(date) + '_' + str(data) + ';')
        dict_pred = {}
        dict_pred['method'] = 'X11_Arima'
        dict_pred['pred_data'] = str_[:-1]
        dict_pred['mre'] = 0.0001
        dict_pred['info'] = time_series['status']
        return dict_pred

    time_series = Do_data.data_for_train(time_series, if_log10=True, zero_ratio=3)

    arima = X11_ARIMA()
    predict = arima.predict_all(time_series, num_preds=pred_date_num)

    explain = Explain(num_preds=pred_date_num, if_pred_all=True, method_str='X11_Arima', date_format=date_format, if_log10=True)
    dict_pred = explain.predict_format(time_series, predict, remaining_data=remaining_data)

    if if_debug:
        plot = plot_table(num_preds=pred_date_num, date_format=date_format, if_log=True)
        plot.plot(time_series, predict)

    return dict_pred

# Prophet
def pred_by_Prophet(csv_path, ColumnDate, ColumnData, date_format, pred_date_num, Do_featuredata, year, month, remaining_data, sql_data):

    Do_data = Do_TimeData(ColumnDate, ColumnData, format=date_format, year=year, month=month, sql_data=sql_data)
    time_series = Do_data.read_time_data(csv_path, limit_num)

    if isinstance(time_series, dict) and time_series['status'] == 'failed':
        return time_series['results']
    elif isinstance(time_series, dict) and time_series['status'] == 'successed':
        predict = [np.divide(remaining_data, pred_date_num)] * pred_date_num
        date_index = time_series['date_index']
        str_ = ''
        for data, date in zip(predict, list(date_index)):
            str_ += (str(date) + '_' + str(data) + ';')
        dict_pred = {}
        dict_pred['method'] = 'Prophet'
        dict_pred['pred_data'] = str_[:-1]
        dict_pred['mre'] = 0.0001
        dict_pred['info'] = time_series['status']
        return dict_pred

    time_series = Do_data.data_for_train(time_series, if_log10=True, zero_ratio=3)
    logger.info('prophet model')
    prophet = Prophetor_modle(num_preds=pred_date_num, date_format=date_format)
    predict = prophet.fitAndPred(time_series)

    explain = Explain(num_preds=pred_date_num, method_str='prophet', date_format=date_format, if_log10=True)
    dict_pred = explain.predict_format(time_series, predict, remaining_data=remaining_data)

    if if_debug:
        plot = plot_table(num_preds=pred_date_num, date_format=date_format)
        plot.plot(time_series, predict)

    return dict_pred


# LightGBM
def pred_by_LightGBM(csv_path, ColumnDate, ColumnData, date_format, pred_date_num, ratio, Do_featuredata, year, month, remaining_data, sql_data):

    feature = lightgbm_holidays(ColumnDate, ColumnData, date_format=date_format, if_log=True, year=year, month=month,
                                sql_data=sql_data)
    new_week_feature, pred_feature, last_year_month = feature.split_features(pred_date_num, csv_path)

    if isinstance(new_week_feature, dict) and new_week_feature['status'] == 'failed':
        return new_week_feature['results']
    elif isinstance(new_week_feature, dict) and new_week_feature['status'] == 'successed':
        predict = [np.divide(remaining_data, pred_date_num)] * pred_date_num
        date_index = new_week_feature['date_index']
        str_ = ''
        for data, date in zip(predict, date_index):
            str_ += (str(date) + '_' + str(data) + ';')
        dict_pred = {}
        dict_pred['method'] = 'LightGBM'
        dict_pred['pred_data'] = str_[:-1]
        dict_pred['mre'] = 1.0
        dict_pred['info'] = new_week_feature['status']
        return dict_pred

    lightgbm = Lightgbm_model(ratio=ratio, date_format=date_format)
    dict_pred = lightgbm.predict_all(new_week_feature, pred_feature, last_year_month)

    return dict_pred

# LSTM
def pred_by_LSTM(csv_path, ColumnDate, ColumnData, date_format, properties_path, sql_data, pred_date_num,
                 step_back, test_ratio, Do_featuredata, year, month, remaining_data):

    do_for_lstm = Do_for_lstm(ColumnDate, ColumnData, date_format, properties_path=properties_path, year=year, month=month, sql_data=sql_data)
    time_series = do_for_lstm.read_time_data(csv_path, limit_num)

    if isinstance(time_series, dict) and time_series['status'] == 'failed':
        return time_series['results']
    elif isinstance(time_series, dict) and time_series['status'] == 'successed':
        predict = [np.divide(remaining_data, pred_date_num)] * pred_date_num
        date_index = time_series['date_index']
        str_ = ''
        for data, date in zip(predict, list(date_index)):
            str_ += (str(date) + '_' + str(data) + ';')
        dict_pred = {}
        dict_pred['method'] = 'LSTM'
        dict_pred['pred_data'] = str_[:-1]
        dict_pred['mre'] = 0.0001
        dict_pred['info'] = time_series['status']
        return dict_pred

    future_time = do_for_lstm.time_for_pred(csv_path, pred_date_num=pred_date_num)
    x_train, y_train, x_test, y_test, all_data, all_label, scaler = do_for_lstm.data_for_train(csv_path,
                                                                                               step_back=step_back,
                                                                                               test_ratio=test_ratio)
    Bilstm = Lstm_model(step_back=step_back, epoch=200, hidden_size=2, num_layer=1, lr=0.02, bidirectional=True)
    predict = Bilstm.predict_all(all_data, all_label, scaler, pred_date_num=pred_date_num)

    explain = Explain(num_preds=pred_date_num, method_str='BiLstm', date_format=date_format,
                      Do_featuredata=Do_featuredata)

    dict_pred = explain.predict_format(time_series, predict, remaining_data=remaining_data, future_time=future_time)
    if if_debug:
        plot = plot_table(num_preds=pred_date_num, date_format=date_format)
        plot.plot(time_series, predict)

    return dict_pred

# TCN
def pred_by_TCN(csv_path, ColumnDate, ColumnData, date_format, pred_date_num, properties_path, sql_data, step_back,
                  test_ratio, Do_featuredata, year, month, remaining_data):

    Do_for_lstm_ = Do_for_lstm(ColumnDate, ColumnData, date_format, properties_path=properties_path, year=year, month=month, sql_data=sql_data)
    time_series = Do_for_lstm_.read_time_data(csv_path, limit_num=limit_num)

    if isinstance(time_series, dict) and time_series['status'] == 'failed':
        return time_series['results']
    elif isinstance(time_series, dict) and time_series['status'] == 'successed':
        predict = [np.divide(remaining_data, pred_date_num)] * pred_date_num
        date_index = time_series['date_index']
        str_ = ''
        for data, date in zip(predict, list(date_index)):
            str_ += (str(date) + '_' + str(data) + ';')
        dict_pred = {}
        dict_pred['method'] = 'LSTM'
        dict_pred['pred_data'] = str_[:-1]
        dict_pred['mre'] = 0.0001
        dict_pred['info'] = time_series['status']
        return dict_pred

    future_time = Do_for_lstm_.time_for_pred(csv_path, pred_date_num=pred_date_num)
    x_train, y_train, x_test, y_test, all_data, all_label, scaler = Do_for_lstm_.data_for_train(csv_path,
                                                                                               step_back=step_back,
                                                                                               test_ratio=test_ratio)
    Tcn = TCN_model(step_back=step_back, lr=0.05, dropout=0.5, kernel_size=2, channel_sizes=4, n_classes=1,
                    input_channels=1, epoch=200)
    predict = Tcn.predict_all(all_data, all_label, scaler, pred_date_num=pred_date_num)

    explain = Explain(num_preds=pred_date_num, method_str='TCN', date_format=date_format,
                      Do_featuredata=Do_featuredata)

    dict_pred = explain.predict_format(time_series, predict, remaining_data=remaining_data, future_time=future_time)
    if if_debug:
        plot = plot_table(num_preds=pred_date_num, date_format=date_format)
        plot.plot(time_series, predict)

    return dict_pred

# 收入预测的python端的入参json：
# --------------------------
# {"incomeType":"11","year":"2021","budgetNum":"9.87654321789E8","month":"5","completedNum":"54321.987","foreMethod":"X11_Arima"}
#
# 支出预测的python端的入参json：
# --------------------------
# {"spendingType":"11","year":"2021","budgetNum":"9.87654321789E8","month":"5","completedNum":"54321.987","foreMethod":"TCN"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treasury_cash")
    parser.add_argument("--foreMethod", type=str, default='LightGBM', choices=["Arima", "X11_Arima", "Prophet", "LightGBM", "LSTM", "TCN"],
                        help="Arima：差分自回归移动平均模型；"
                             "X11_Arima：基于arima的季节趋势模型；"
                             "Prophet：prophet模型；"
                             "LightGBM：基于树的模型；"
                             "LSTM：长短时模型；"
                             "TCN：时间卷积模型")
    parser.add_argument("--year", type=int, default=2020,
                        help="预测年份")

    parser.add_argument("--budgetNum", type=int, default=28715693112,
                        help="年初预算总值")
    parser.add_argument("--month", type=int, default=3,
                        help="已完成月份")
    parser.add_argument("--completedNum", type=int, default=9571897704,
                        help="已完成总值")
    args = parser.parse_args()

    pred_Treasury_cash = Treasury_cash(foreMethod=args.foreMethod, incomeType=args.incomeType, year=args.year, budgetNum=args.budgetNum, month=args.month, completedNum=args.completedNum)

    dict_pred = pred_Treasury_cash.choose_algorithm()

    logger.info('the result of predict is {}'.format(dict_pred))
