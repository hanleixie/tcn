# -*- coding:utf-8 -*-
# @Time: 2020/11/13 11:01
# @File: server.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequestKeyError
import platform
import json
from Treasury_cash.pred_model import *
from common.log import logger, init_logger
init_logger()#log_path)
logger.info('==>start')
import time
t = time.time()
app = Flask('app')

from Treasury_cash.pred_model import *

# 国库现金收入
@app.route('/Treasure_cash', methods=['POST'])
def apicall_1():
    try:
        start = time.time()
        req_data = request.get_data(as_text=True)
        logger.info(req_data)
        if req_data:
            req_data = json.loads(req_data)

            foreMethod = req_data['foreMethod']
            logger.info('<<{}'.format(foreMethod))
            incomeType = req_data['incomeType']
            year = req_data['year']
            budgetNum = req_data['budgetNum']
            month = req_data['month']
            completedNum = req_data['completedNum']

            pred_Treasury_cash = Treasury_cash(foreMethod=foreMethod, incomeType=incomeType, year=int(year), budgetNum=int(budgetNum), month=int(month), completedNum=int(completedNum))
            dict_pred = pred_Treasury_cash.choose_algorithm()

            logger.info("时间{}".format(time.time() - start))
            dict_ = {}
            dict_['status'] = 'success'
            dict_['results'] = dict_pred

            return jsonify(dict_)
        else:
            res = {'status': 'failed', 'results': '没有收到request消息'}
            return jsonify(res)
    except BadRequestKeyError as e:
        logger.error(e)
        res = {'status': 'failed', 'results': str(e)}
        return jsonify(res)
    except FileNotFoundError as e:
        logger.error(e)
        res = {'status': 'failed', 'results': e.strerror}
        return jsonify(res)
    except Exception as e:
        logger.error(e)
        res = {'status': 'failed', 'results': str(e)}
        return jsonify(res)


@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
    }
    resp = jsonify(message)
    resp.status_code = 400
    return resp



if __name__ == "__main__":

    properties_path = father_path + "/conf/" + "oracle_server.properties"
    prop = Properties(properties_path)
    dict_1 = prop.getProperties()
    server_port = dict_1['server']['port']

    logger.info('<<<Ocr Server Started')
    sysstr = platform.system()
    logger.info(sysstr)
    if (sysstr == "Windows"):
        app.run(host="0.0.0.0", port=int(server_port), threaded=False)
    elif (sysstr == "Linux"):
        app.run(host="0.0.0.0", port=int(server_port), threaded=True)
    else:
        app.run(host="0.0.0.0", port=int(server_port), threaded=True)
    logger.info('<<<Ocr Server stopped')