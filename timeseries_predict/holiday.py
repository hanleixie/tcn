# -*- coding:utf-8 -*-
# @Time: 2020/10/30 10:11
# @File: holiday.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
# 节假日简单规则：
# 1、元旦：{
#         1、星期一和星期五的话和周末连休；
#         2、星期二和星期四的话放三天，补一天
#         3、星期三放一天
#         4、星期六和星期天的话和周一补休
#         }
# 2、春节：一般放假七天，前后各补一天班，一般在上周六和下周日各一天，农历除夕开始
# 3、清明节：四月五日左右，规则同元旦
# 4、劳动节：五月一日，规则同元旦
# 5、端午节：农历五月五日，规则同元旦
# 6、中秋节：农历八月十五，规则同元旦
# 7、国庆节：{
#           1、单独放假不和中秋放一起，规则同春节；
#           2、和中秋混在一起放：
#                            1、放八天，如果截止日期为周日则只补国庆前一个周末的某天
#                            2、放八天，八号截止前后每周末各补一天
#           }
import pandas as pd
from datetime import datetime, date, timedelta
from zhdate import ZhDate
from log import logger, init_logger
init_logger()#log_path)
logger.info('==>start')

start_time = '20200101'
end_time = '20201231'
all_time = [x.date() for x in pd.date_range(start_time, end_time)]


class Holiday():

    def __init__(self, year):
        '''
        :param
        :param
        :param
        '''
        self.year = year


    def yuandan(self):
        year, month, day = str(self.year), '01', '01'
        return self.some_holiday(year, month, day)

    def chunjie(self):
        lunar = ZhDate(self.year, 1, 1)
        solar = lunar.to_datetime()
        #solar = pd.date_range(str(solar).split(' ')[0], periods=1)
        str_solar = str(solar).split(' ')[0]
        year, month, day = str_solar.split('-')[0], str_solar.split('-')[1], str_solar.split('-')[-1]
        chuxi_solar = datetime(int(year), int(month), int(day)) + timedelta(days=-1)
        # chuxi_time, make_chuxi_time = self.Spring_Festival(chuxi_solar)
        return self.Spring_Festival(chuxi_solar)

    def qingming(self):
        # lunar = ZhDate(self.year, 4, 5)  # 新建农历 2010年正月初一 的日期对象
        # solar = lunar.to_datetime()
        # #solar = pd.date_range(str(solar).split(' ')[0], periods=1)
        # str_solar = str(solar).split(' ')[0]
        # year, month, day = str_solar.split('-')[0], str_solar.split('-')[1], str_solar.split('-')[-1]
        year, month, day = str(self.year), '04', '05'
        return self.some_holiday(year, month, day)

    def laodong(self):
        year, month, day = str(self.year), '05', '01'
        return self.some_holiday(year, month, day)

    def duanwu(self):
        lunar = ZhDate(self.year, 5, 5)  # 新建农历 2010年正月初一 的日期对象
        solar = lunar.to_datetime()
        # solar = pd.date_range(str(solar).split(' ')[0], periods=1)
        str_solar = str(solar).split(' ')[0]
        year, month, day = str_solar.split('-')[0], str_solar.split('-')[1], str_solar.split('-')[-1]
        return self.some_holiday(year, month, day)

    def zhongqiu(self):
        lunar = ZhDate(self.year, 8, 15)  # 新建农历 2010年正月初一 的日期对象
        solar = lunar.to_datetime()
        # solar = pd.date_range(str(solar).split(' ')[0], periods=1)
        str_solar = str(solar).split(' ')[0]
        year, month, day = str_solar.split('-')[0], str_solar.split('-')[1], str_solar.split('-')[-1]
        return self.some_holiday(year, month, day)

    def guoqing(self):
        # lunar = ZhDate(self.year, 1, 1)
        # solar = lunar.to_datetime()
        # #solar = pd.date_range(str(solar).split(' ')[0], periods=1)
        # str_solar = str(solar).split(' ')[0]
        # year, month, day = str_solar.split('-')[0], str_solar.split('-')[1], str_solar.split('-')[-1]
        # chuxi_solar = datetime(int(year), int(month), int(day)) + timedelta(days=-1)
        # # chuxi_time, make_chuxi_time = self.Spring_Festival(chuxi_solar)
        year_month_day = str(self.year) + '10' + '01'
        return self.Spring_Festival(year_month_day)


    def some_holiday(self, year, month, day):
        '''
        :param year:input year
        :return:
        some_holiday_time:holidays time
        make_time:work overtime
        '''
        global time_, make_
        some_holiday_time = []
        make_time = []
        make_ = []
        year_time = pd.to_datetime(year + month + day, format='%Y%m%d')
        week_num = year_time.isoweekday()
        logger.info('{}是周{}'.format(year + month + day, week_num))
        if week_num == int(3):
            time_ = pd.date_range(year + month + day, periods=1)
        elif week_num == int(1) or week_num == int(5) or week_num == int(6):
            if week_num == int(5) or week_num == int(6):
                time_ = pd.date_range(year + month + day, periods=3)
            else:
                offset = datetime(int(year), int(month), int(day)) + timedelta(days=-2)
                time_ = pd.date_range(offset, periods=3)
        elif week_num == int(7):
            time_ = pd.date_range(year + month + day, periods=2)
        elif week_num == int(4):
            time_ = pd.date_range(year + month + day, periods=3)
            offset = datetime(int(year), int(month), int(day)) + timedelta(days=3)
            make_ = pd.date_range(offset, periods=1)
        else:
            offset = datetime(int(year), int(month), int(day)) + timedelta(days=-2)
            time_ = pd.date_range(offset, periods=3)
            offset_ = datetime(int(year), int(month), int(day)) + timedelta(days=-3)
            make_ = pd.date_range(offset_, periods=1)

        some_holiday_time.extend([x.strftime('%F') for x in time_])
        make_time.extend([x.strftime('%F') for x in make_ if x])
        return some_holiday_time, make_time

    def Spring_Festival(self, chuxi_solar):

        chuxi_time = []  # 假期
        make_chuxi_time = []  # 加班
        make_ago = None
        chuxi_solar_time = pd.to_datetime(chuxi_solar, format='%Y%m%d')
        week_num = chuxi_solar_time.isoweekday()
        logger.info('{}是周{}'.format(chuxi_solar_time, week_num))
        str_solar = str(chuxi_solar_time).split(' ')[0]
        year, month, day = str_solar.split('-')[0], str_solar.split('-')[1], str_solar.split('-')[-1]
        if week_num in [2, 3, 4, 5]:
            offset_ago = datetime(int(year), int(month), int(day)) + timedelta(days=-week_num)
            make_ago = pd.date_range(offset_ago, periods=1)
            make_chuxi_time.extend([x.strftime('%F') for x in make_ago])

            offset_after = datetime(int(year), int(month), int(day)) + timedelta(days=(13 - week_num))
            make_after = pd.date_range(offset_after, periods=1)
            make_chuxi_time.extend([x.strftime('%F') for x in make_after])

            time_ = pd.date_range(chuxi_solar_time, periods=7)
            chuxi_time.extend([x.strftime('%F') for x in time_])

        elif week_num == int(7):
            offset_ago = datetime(int(year), int(month), int(day)) + timedelta(days=-1)
            make_ago = pd.date_range(offset_ago, periods=1)
            make_chuxi_time.extend([x.strftime('%F') for x in make_ago])

            offset_after = datetime(int(year), int(month), int(day)) + timedelta(days=7)
            make_after = pd.date_range(offset_after, periods=1)
            make_chuxi_time.extend([x.strftime('%F') for x in make_after])

            time_ = pd.date_range(chuxi_solar_time, periods=7)
            chuxi_time.extend([x.strftime('%F') for x in time_])
        elif week_num == int(6):
            offset_after = datetime(int(year), int(month), int(day)) + timedelta(days=7)
            make_after = pd.date_range(offset_after, periods=2)
            make_chuxi_time.extend([x.strftime('%F') for x in make_after])

            time_ = pd.date_range(chuxi_solar_time, periods=7)
            chuxi_time.extend([x.strftime('%F') for x in time_])
        else:
            offset_before = datetime(int(year), int(month), int(day)) + timedelta(days=-2)
            make_after = pd.date_range(offset_before, periods=2)
            make_chuxi_time.extend([x.strftime('%F') for x in make_after])

            time_ = pd.date_range(chuxi_solar_time, periods=7)
            chuxi_time.extend([x.strftime('%F') for x in time_])


        return chuxi_time, make_chuxi_time


def all_holiday(year):
    holidays = []
    work_overtime = []
    holiday = Holiday(year=year)
    # yuandan
    some_holiday_time_yuandan, make_time_yuandan = holiday.yuandan()
    holidays.extend(some_holiday_time_yuandan)
    work_overtime.extend(make_time_yuandan)
    # holidays.extend([x.strftime('%F')[0] for x in some_holiday_time_yuandan if x[0]])
    # if make_time_yuandan != [None]:
    #     work_overtime.extend([x.strftime('%F')[0] for x in make_time_yuandan])
    # chunjie
    some_holiday_time_chunjie, make_time_chunjie = holiday.chunjie()
    holidays.extend(some_holiday_time_chunjie)
    work_overtime.extend(make_time_chunjie)
    # print(some_holiday_time_chunjie)
    # print(make_time_chunjie)
    # holidays.extend([x.strftime('%F')[0] for x in some_holiday_time_chunjie])
    # if make_chuxi_time_chujie != [None]:
    #     work_overtime.extend([x.strftime('%F')[0] for x in make_chuxi_time_chujie])
    # qingming
    some_holiday_time_qingming, make_time_qingming = holiday.qingming()
    holidays.extend(some_holiday_time_qingming)
    work_overtime.extend(make_time_qingming)
    # holidays.extend([x.strftime('%F')[0] for x in some_holiday_time_qingming if x[0]])
    # if make_time_qingming != [None]:
    #     work_overtime.extend([x.strftime('%F')[0] for x in make_time_qingming])
    # laodong
    some_holiday_time_laodong, make_time_laodong = holiday.laodong()
    holidays.extend(some_holiday_time_laodong)
    work_overtime.extend(make_time_laodong)
    # duanwu
    some_holiday_time_duanwu, make_time_duanwu = holiday.duanwu()
    holidays.extend(some_holiday_time_duanwu)
    work_overtime.extend(make_time_duanwu)
    # zhongqiu
    some_holiday_time_zhongqiu, make_time_zhongqiu = holiday.zhongqiu()

    holidays.extend(some_holiday_time_zhongqiu)
    work_overtime.extend(make_time_zhongqiu)
    # guoqing
    some_holiday_time_guoqing, make_time_guoqing = holiday.guoqing()
    holidays.extend(some_holiday_time_guoqing)
    work_overtime.extend(make_time_guoqing)

    start_ = str(year) + '01' + '01'
    end_ = str(year) + '12' + '31'
    word_days = pd.bdate_range(start_, end_, freq='b')
    work_days = [x.strftime('%F') for x in word_days]
    work_days.extend([x for x in work_overtime if x not in work_days])
    for x in holidays:
        if x in work_days:
            work_days.remove(x)

    return work_days

work_days = all_holiday(2019)