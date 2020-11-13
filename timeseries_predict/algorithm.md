# ALGORITHM

## 1、目的

* 通过点选界面获得输入信息，选择合适的算法完成要求预测年份截止到已完成月份之后的未来月份数据。

* 给定输入：1、年初预算总值；2、收入类型；3、预测年份；4、已完成月份；5、已完成数值；6、算法选择。

  *收入类型：11：一般公共预算、12：政府性基金、13：国有资本经营*

* 期望输出：1、method；2、预测月份和预测数据；3、评价指标；4、info信息。

## 2、项目组件

**数据处理**

* 数据处理类方法包括对时间数据的处理，满足 arima、X11_arima 和 prophet 的输入要求，详见`TaxRevenueForecast/common/Data_Preprocess.py`中`Do_TimeData()`类方法
* 对时间数据进行拆分，将其拆分为年、月、日、第几周，周几和第几季度，满足 lightgbm 的输入要求，详见`TaxRevenueForecast/common/Data_Preprocess.py`中`lightgbm_holidays()`类方法
* 对时间数据进行回溯处理，通过未来几天的数据预测当天的数据，满足 LSTM 和 TCN 的输入要求，详见`TaxRevenueForecast/common/Data_Preprocess.py`中`Do_for_lstm()`类方法

**模型算法**

* Arima算法：差分自回归移动平均模型，详见`TaxRevenueForecast/Treasury_cash/models.py`中`Arima_model()`类方法
* X11_arima算法：详见`TaxRevenueForecast/Treasury_cash/models.py`中`X11_Arima()`类方法
* prophet算法：详见`TaxRevenueForecast/Treasury_cash/models.py`中`Prophet_model()`类方法
* lightgbm算法：详见`TaxRevenueForecast/Treasury_cash/models.py`中`Lightgbm_model()`类方法
* Lstm算法：详见`TaxRevenueForecast/Treasury_cash/models.py`中`Lstm_model()`类方法，本项目使用双向的 lstm 算法
* TCN算法：详见`TaxRevenueForecast/Treasury_cash/models.py`中`TCN_model()`类方法

## 3、调用方法

* 整体算法的调用在`TaxRevenueForecast/Treasury_cash/pred_model.py`中`Treasury_cash()`类方法
* 通过修改传入参数的类型完成不同算法的预算，运行`python pred_model.py`

## 4、注意事项

* arima 和 lightgbm在预测时对数据量大小要求很高，小数据量时预测值为同一数值，没有使用意义。
* 在 lightgbm 使用过程中只支持通过对日的预测，然后将每月的预测值加和，同时去掉每个月的节假日和周末。
* 在所有的算法中当数据量特别少不具有使用算法预测的资格时，返回年初预算总值和已完成数值的差值的平均值。
* 当要预测月份的和不与年初预算总值和已完成数值的差值相等时，等比例的对月预测数据进行补偿，比例系数为每个月占所有预测月份的百分比。
* 支持月和日的预测，但对年的预测没有进行测试。
* 本项目添加了节假日信息，对日的预测时去掉预测日中属于节假日和周末的预测。
* 节假日和周末计算中，去除国家在某一年突然决定在某个节假日多放假的天数和中秋与国庆重合放假的天数，其余正常年份计算出来的天数皆很准确。详见`TaxRevenueForecast/Treasury_cash/holiday.py`
* 在处理时间序列数据时，对arima、X11_arima、prophet和lightgbm的输入数据进行了log变换，对lstm和tcn的输入数据进行归一化处理。
* 当给传入的已完成月份为n时，读取数据库却只有月份n之前的数据，缺少n月份数据，此时返回status：failed，results：数据库缺少到已完成月份 n 前的部分历史数据。
* 关于日志信息，当init_logger()中传入log_path时，日志信息将被保存。


