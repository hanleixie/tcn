# -*- coding:utf-8 -*-
# @Time: 2020/10/22 10:09
# @File: config.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------

# 日志保存地址
log_path = 'logs/logs.txt'
# 配置文件目录
properties_path = 'conf/oracle_server.properties'
# csv文件目录
# csv_path = r'E:\yonyou\newsdata\all_month_both.csv'
csv_path = None
# csv_path = r'C:\Users\Administrator\PycharmProjects\one\yonyou\Treasury_cash\data\day_data.csv'

# 最少数据个数值
limit_num = 12
# 选用那两列
# ColumnDate = 'RQ'
# ColumnData = 'RK'
ColumnDate = 'YEARMONTH'
ColumnData = 'MONEY'
# ColumnDate = 'YEARMONTHDAY'
# ColumnData = 'SUM(MONEY)'
# 日期格式
date_format = '%Y%m'
# date_format = '%Y%m%d'
# 11类型
sql_data_11 = 'month_data_11'

incomeType = "11"
year = 2021
budgetNum = 9.87654321789E8
month = 5
completedNum = 54321.987
foreMethod = 'X11_Arima'

# gbm预测时间长度
pred_date_num = 8
# 切分日期作为特征
Do_featuredata = True
# 测试集比例
ratio = 0.4
# lstm回溯的步数
step_back = 3
# 训练集比例
test_ratio = 0.2








