# analyses results from mainer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import random
import json
import threading
from datetime import datetime, timedelta
import time
import os
plt.style.use('dark_background')
import re

a_line = '--------------------------------------------------------------------------'
print('ok1')

BASE = 'AUD'

# 24.816454850188563
# 25.259852574115328
# top 26
# average first sell diff %: 0.09211538461538461
# average buy diff %: -0.39538461538461533
# top 17
# average first sell diff %: 0.095
# average buy diff %: -0.3911764705882353
# top 12
# average first buy diff %: -0.38500000000000006
# average sell diff %: 0.09541666666666666
BUY_LEVELS = [1, 2, 3]
TRADE_STRATEGIES = ['percent', 'percent off ma', 'std', 'std off ma']

def parse(s):
    if 'day' in s:
        m = re.match(r'(?P<days>[-\d]+) day[s]*, (?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)', s)
    else:
        m = re.match(r'(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d[\.\d+]*)', s)
    return {key: float(val) for key, val in m.groupdict().items()}

# getting the results sorted by EUR profit from their files
results = []
result_dir = 'results/'
file_names = os.listdir(result_dir)
res_count = len(file_names)
for i in range(res_count):
    results.append(json.load(open(f'results/result{i}.json')))
print('ok2')
# getting stats on tests that pass filter
filter_count = 0
filter_params = {}
for strat in TRADE_STRATEGIES:
        filter_params[f'buy: {strat}'] = 0
        filter_params[f'sell: {strat}'] = 0
for buy_strat in TRADE_STRATEGIES:
    for sell_strat in TRADE_STRATEGIES:
        filter_params[f'{buy_strat}/{sell_strat}'] = 0
for level_count in BUY_LEVELS:
    filter_params[level_count] = 0
filter_params['cum usd profit'] = 0
filter_params['cum sell p'] = 0
filter_params['cum first buy p'] = 0
filter_params['buys'] = 0
filter_params['sells'] = 0
filter_params['cum drawdown'] = 0
for i in range(res_count):
    # filter = results[i]['profit percent']['EUR'] >= 24.816
    filter = results[i]['profit percent'][BASE] >= 5
    #filter = results[i]['profit percent']['USD'] >= 18 # top 103
    #filter = results[i]['profit percent']['USD'] >= 16
    #filter = results[i]['max drawdown']['EUR'] < 1
    #filter = res_e_p[i]['params']['amount buy levels'] > 1
    #filter = res_e_p[i]['params']['amount buy levels'] == 3
    #filter = res_e_p[i]['params']['buy strat'] == 'percent'
    #filter = res_e_p[i]['params']['sell strat'] == 'std off ma'
    #filter = timedelta(**parse(res_e_p[i]['backtest duration'])) >= timedelta(minutes = 1)
    if filter:
        filter_count += 1
        filter_params[f"{results[i]['params']['buy strat']}/{results[i]['params']['sell strat']}"] += 1
        filter_params[f"buy: {results[i]['params']['buy strat']}"] += 1
        filter_params[f"sell: {results[i]['params']['sell strat']}"] += 1
        filter_params['cum usd profit'] += results[i]['profit percent']['USD']
        filter_params[results[i]['params']['amount buy levels']] += 1
        filter_params['buys'] += results[i]['trade count']['buys']
        filter_params['sells'] += results[i]['trade count']['sells']
        filter_params['cum drawdown'] += results[i]['max drawdown']['USD']
        if results[i]['params']['buy strat'] == 'percent' and results[i]['params']['sell strat'] == 'percent':
            filter_params['cum sell p'] += results[i]['params']['sell percent']
            filter_params['cum first buy p'] += results[i]['params']['buy percents'][0]
print(a_line)
print(f'Out of {res_count}, {filter_count} pass the filter, that\'s {round(100*filter_count/res_count, 2)}%')
print(a_line)
avg_buy_p = filter_params['cum first buy p']/filter_count
avg_sell_p = filter_params['cum sell p']/filter_count
avg_sell_count = filter_params['sells'] / filter_count
avg_buy_count = filter_params['buys'] / filter_count
avg_usd_profit = filter_params['cum usd profit'] / filter_count
avg_usd_drawdown = filter_params['cum drawdown'] / filter_count
print('Trade strategy and the percentage of strategies that used it\n' + a_line)
print('strategy |percent        |percent off ma |std            |std off ma     |\n' + a_line)
str_buy = 'buy      |'
str_sell = 'sell     |'
for strat in TRADE_STRATEGIES:
    buy_strat_p = str(round(100 * filter_params[f'buy: {strat}'] / filter_count, 2)) + '%'
    sell_strat_p = str(round(100 * filter_params[f'sell: {strat}'] / filter_count, 2)) + '%'
    buy_str_fill = 15 - len(buy_strat_p)
    sell_str_fill = 15 - len(sell_strat_p)
    str_buy += buy_strat_p + buy_str_fill * ' ' + '|'
    str_sell += sell_strat_p + sell_str_fill * ' ' + '|'
print(str_buy)
print(str_sell)
print(a_line)
print(f'average first buy diff %: {avg_buy_p}')
print(f'average sell diff %: {avg_sell_p}')
print(f'average USD profit: {avg_usd_profit}')
print(f'average USD drawdown: {avg_usd_drawdown}')
print(f'average buy count: {avg_buy_count}')
print(f'average sell count: {avg_sell_count}')
print(a_line)
print('Buy/Sell strategy combinations (does\'nt show 0%)')
for buy_strat in TRADE_STRATEGIES:
    for sell_strat in TRADE_STRATEGIES:
        percentage = round(100 * filter_params[f'{buy_strat}/{sell_strat}'] / filter_count, 2)
        if percentage:
            print(f'{buy_strat}/{sell_strat}" : {percentage}%')
print(a_line)
print('Stats for buy level amount: ')
for level_count in BUY_LEVELS:
    level_count_p = round(100 * filter_params[level_count] / filter_count, 2)
    print(f'{level_count} buy level count strategies: {level_count_p}%')

'''
{
    "params": {
        "buy strat": "percent",
        "sell strat": "percent",
        "amount buy levels": 1,
        "buy lots": [
            100
        ],
        "buy percents": [
            -0.5
        ],
        "sell percent": 0
    },
    "profit percent": {
        "EUR": 25.751876181841737,
        "USD": 20.364941761439567
    },
    "max drawdown": {
        "EUR": 0.10024384264006164,
        "USD": 0.1336187268916974
    },
    "trade count": {
        "buys": 48,
        "sells": 48,
        "long trades": 48,
        "short trades": 47
    },
    "win rate": {
        "overall": 94.73684210526315,
        "long wins": 47,
        "short wins": 43
    },
    "avg duration": {
        "long trade": "0 days 21:53:03.750000",
        "short trade": "1 days 11:50:52.340425531"
    },
    "backtest duration": "0:00:16.911774",
}
'''