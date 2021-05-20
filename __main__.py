# backtests and optimises eur price arbitrage against stable coins and usd
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

plt.style.use('dark_background')
from datetime import datetime, timedelta

DIR_NAME = '2021data'

FOREX_SYMBOLS = ['EURUSD']
STABLE_SYMBOLS = ['EURUSDT']

DAYMINS = 1440

# backtest params
START_USD_BALANCE = 100

'''EMA_LENGTHS = [DAYMINS * 7, DAYMINS/2, DAYMINS * 14]

# sell price parameters
SELL_DIFFS = [-0.2, -0.15, -0.1, -0.05, -0.02, 0, 0.05,  0.1, 0.15, 0.2]
SELL_STDS = [0.3, 0.4, 0.5, 0.6, 0.7, 1, 1.2, 1.5]

BUY_LEVELS = [1, 2]

# percentages
BUY_DIFFS = {}
BUY_DIFFS[1] = [[-0.4], [-0.45], [-0.5], [-0.55], [-0.6], [-0.65], [-0.7]]
BUY_DIFFS[2] = [[-0.4, -1], [-0.5, -1], [-0.5, -0.7], [-0.4, -0.7], [-0.4, -0.5], [-0.7, -0.8]]
# BUY_DIFFS[3] = [[-0.4, -0.7, -1.1]]

# standard deviations for entry points
BUY_STDS = {}
BUY_STDS[1] = [[-1], [-2], [-3]]
BUY_STDS[2] = [[-2, -3], [-1, -2]]
# BUY_STDS[3] = [[-1, -2, -3]]

# buy lots if there will be multiple entry points
BUY_LOTS = {}
BUY_LOTS[1] = [[100]]
BUY_LOTS[2] = [[50, 50], [33, 67], [67, 33], [25, 75], [75, 25], [60, 40], [40, 60]]
#BUY_LOTS[3] = [[33, 33, 34], [25, 25, 50], [50, 25, 25]]

TRADE_STRATEGIES = ['percent', 'percent off ma', 'std off ma', 'std']
BUY_STRATEGIES = ['percent', 'percent off ma', 'std off ma', 'std']'''

EMA_LENGTHS = [DAYMINS * 3, DAYMINS/2, DAYMINS/4, DAYMINS*7 ]

# sell price parameters
SELL_DIFFS = [-0.07, 0.075, 0.08, 0.085, 0.09, 0.925, 0.095,  0.1, 0.105, 0.11, 0.115, 0.12, 0.125]
# SELL_STDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.2]

BUY_LEVELS = [1]

# percentages
BUY_DIFFS = {}
BUY_DIFFS[1] = [[-0.35], [-0.37],[-0.385], [-0.39], [-0.4], [-0.41], [-0.43], [-0.45],  [-0.47],  [-0.49],  [-0.5],  [-0.51],  [-0.53],  [-0.55]]
# BUY_DIFFS[2] = [[-0.4, -1], [-0.5, -1], [-0.5, -0.7], [-0.4, -0.7], [-0.4, -0.5], [-0.7, -0.8]]
# BUY_DIFFS[3] = [[-0.4, -0.7, -1.1]]

# standard deviations for entry points
BUY_STDS = {}
# BUY_STDS[1] = [[-2.5], [-2], [-3], [-3.5], [-2.2], [-1.8]]
# BUY_STDS[2] = [[-2, -3], [-1, -2]]
# BUY_STDS[3] = [[-1, -2, -3]]

# buy lots if there will be multiple entry points
BUY_LOTS = {}
BUY_LOTS[1] = [[100]]
#BUY_LOTS[2] = [[50, 50], [33, 67], [67, 33], [25, 75], [75, 25], [60, 40], [40, 60]]
#BUY_LOTS[3] = [[33, 33, 34], [25, 25, 50], [50, 25, 25]]

TRADE_STRATEGIES = ['percent']
BUY_STRATEGIES = ['percent']

MAX_THREADS = multiprocessing.cpu_count()

# binance fee when using BNB = 0.00075
FEE = 0.0006

LINE = '-----------------------------------------------------------------------'

global param_len
global finished_tests
finished_tests = 0

def params_tostring(p_comb):
    return json.dumps(p_comb, indent = 4)

def get_param_combinations():
    param_combinations = []
    param_comb = {}
    for buy_strat in BUY_STRATEGIES:
        for sell_strat in TRADE_STRATEGIES:
            for amount_levels in BUY_LEVELS:
                    for buy_lots in BUY_LOTS[amount_levels]:
                        if buy_strat == 'percent' or buy_strat == 'percent off ma':
                            for buy_diffs in BUY_DIFFS[amount_levels]:
                                if sell_strat == 'percent' or sell_strat == 'percent off ma':
                                    for sell_diff in SELL_DIFFS:
                                        if (sell_strat == 'percent off ma') or (buy_strat == 'percent off ma') or (sell_strat == 'std') or (buy_strat == 'std'):
                                            for ema_length in EMA_LENGTHS:
                                                param_comb['ma length'] = ema_length
                                                param_comb['buy strat'] = buy_strat
                                                param_comb['sell strat'] = sell_strat
                                                param_comb['amount buy levels'] = amount_levels
                                                param_comb['buy lots'] = buy_lots
                                                param_comb['buy percents'] = buy_diffs
                                                param_comb['sell percent'] = sell_diff
                                                param_combinations.append(param_comb)
                                                param_comb = {}
                                        else:
                                            param_comb['buy strat'] = buy_strat
                                            param_comb['sell strat'] = sell_strat
                                            param_comb['amount buy levels'] = amount_levels
                                            param_comb['buy lots'] = buy_lots
                                            param_comb['buy percents'] = buy_diffs
                                            param_comb['sell percent'] = sell_diff
                                            param_combinations.append(param_comb)
                                            param_comb = {}
                                elif sell_strat == 'std' or sell_strat == 'std off ma':
                                    for sell_std in SELL_STDS:
                                        if (sell_strat == 'std off ma') or (buy_strat == 'percent off ma') or (sell_strat == 'std') or (buy_strat == 'std'):
                                            for ema_length in EMA_LENGTHS:
                                                param_comb['ma length'] = ema_length
                                                param_comb['buy strat'] = buy_strat
                                                param_comb['sell strat'] = sell_strat
                                                param_comb['amount buy levels'] = amount_levels
                                                param_comb['buy lots'] = buy_lots
                                                param_comb['buy percents'] = buy_diffs
                                                param_comb['sell std'] = sell_std
                                                param_combinations.append(param_comb)
                                                param_comb = {}
                                        else:
                                            param_comb['buy strat'] = buy_strat
                                            param_comb['sell strat'] = sell_strat
                                            param_comb['amount buy levels'] = amount_levels
                                            param_comb['buy lots'] = buy_lots
                                            param_comb['buy percents'] = buy_diffs
                                            param_comb['sell std'] = sell_std
                                            param_combinations.append(param_comb)
                                            param_comb = {}
                        elif buy_strat == 'std' or buy_strat ==  'std off ma':
                            for buy_stds in BUY_STDS[amount_levels]:
                                if sell_strat == 'percent' or sell_strat == 'percent off ma':
                                    for sell_diff in SELL_DIFFS:
                                        if (sell_strat == 'percent off ma') or (buy_strat ==  'std off ma') or (sell_strat == 'std') or (buy_strat == 'std'):
                                            for ema_length in EMA_LENGTHS:
                                                param_comb['ma length'] = ema_length
                                                param_comb['buy strat'] = buy_strat
                                                param_comb['sell strat'] = sell_strat
                                                param_comb['amount buy levels'] = amount_levels
                                                param_comb['buy lots'] = buy_lots
                                                param_comb['buy stds'] = buy_stds
                                                param_comb['sell percent'] = sell_diff
                                                param_combinations.append(param_comb)
                                                param_comb = {}
                                        else:
                                            param_comb['buy strat'] = buy_strat
                                            param_comb['sell strat'] = sell_strat
                                            param_comb['amount buy levels'] = amount_levels
                                            param_comb['buy lots'] = buy_lots
                                            param_comb['buy stds'] = buy_stds
                                            param_comb['sell percent'] = sell_diff
                                            param_combinations.append(param_comb)
                                            param_comb = {}
                                elif sell_strat == 'std' or sell_strat == 'std off ma':
                                    for sell_std in SELL_STDS:
                                        if (sell_strat == 'std off ma') or (buy_strat == 'std off ma') or (sell_strat == 'std') or (buy_strat == 'std'):
                                            for ema_length in EMA_LENGTHS:
                                                param_comb['ma length'] = ema_length
                                                param_comb['buy strat'] = buy_strat
                                                param_comb['sell strat'] = sell_strat
                                                param_comb['amount buy levels'] = amount_levels
                                                param_comb['buy lots'] = buy_lots
                                                param_comb['buy stds'] = buy_stds
                                                param_comb['sell std'] = sell_std
                                                param_combinations.append(param_comb)
                                                param_comb = {}
                                        else:
                                            param_comb['buy strat'] = buy_strat
                                            param_comb['sell strat'] = sell_strat
                                            param_comb['amount buy levels'] = amount_levels
                                            param_comb['buy lots'] = buy_lots
                                            param_comb['buy stds'] = buy_stds
                                            param_comb['sell std'] = sell_std
                                            param_combinations.append(param_comb)
                                            param_comb = {}
    return param_combinations

# populates price data dict
def populate_data_dict():
    data = {}
    file_path = f'{DIR_NAME}/EURUSD_1m.csv'
    data['EURUSD'] = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    for stable_symbol in STABLE_SYMBOLS:
        file_path = f'{DIR_NAME}/{stable_symbol}_1m.csv'
        data[stable_symbol] = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    return data

# adds all possible indicators that the strategy might need
def add_indicators(data):
    data['EURUSDT']['diff'] = ((data['EURUSDT']['close'] - data['EURUSD']['close'])/data['EURUSD']['close']) * 100
    for ema_length in EMA_LENGTHS:
        data['EURUSDT'][f'EMA{ema_length}'] = data['EURUSDT']['diff'].ewm(span=ema_length, adjust=False).mean()
        data['EURUSDT'][f'STD{ema_length}'] = data['EURUSDT']['diff'].ewm(ema_length).std()
    return data

# backtests strategy and returns results
def backtest(df, params):
    backtest_start_time = datetime.now()
    result = {'params': params}
    buy_levels = list(range(1, params['amount buy levels']+1))
    if params['sell strat'] == 'percent':
        df['sell signal'] = df['diff'] > params['sell percent']
    elif params['sell strat'] == 'percent off ma':
        df['sell signal'] = df['diff'] > df[f'EMA{params["ma length"]}'] + params['sell percent']
    elif params['sell strat'] == 'std':
        df['sell signal'] = df['diff'] > df[f'STD{params["ma length"]}'] * params['sell std']
    elif params['sell strat'] == 'std off ma':
        df['sell signal'] = df['diff'] > df[f'STD{params["ma length"]}'] * params['sell std'] + df[f'EMA{params["ma length"]}']
    for buy_level in buy_levels:
        if params['buy strat'] == 'percent':
            df[f'buy signal{buy_level}'] = df['diff'] < params[f'buy percents'][buy_level-1]
        elif params['buy strat'] == 'percent off ma':
            df[f'buy signal{buy_level}'] = df['diff'] < df[f'EMA{params["ma length"]}'] + params[f'buy percents'][buy_level-1]
        elif params['buy strat'] == 'std':
            df[f'buy signal{buy_level}'] = df['diff'] < df[f'STD{params["ma length"]}'] * params[f'buy stds'][buy_level-1]
        elif params['buy strat'] == 'std off ma':
            df[f'buy signal{buy_level}'] = df['diff'] < df[f'STD{params["ma length"]}'] * params[f'buy stds'][buy_level-1] + df[f'EMA{params["ma length"]}']
        # initialising price columns
        df[f'buy price{buy_level}'] = np.nan
    # in order to calculate win rate we will have an additional column
    df['avg buy price'] = np.nan
    df['sell price'] = np.nan
    df['EUR balance'] = np.nan
    df['USD balance'] = np.nan
    usd_bal = START_USD_BALANCE
    usd_bal_before_any_eur_entries = usd_bal
    eur_start_value = START_USD_BALANCE / df['close'].iloc[0]
    df.iloc[0, df.columns.get_loc('USD balance')] = usd_bal
    df.iloc[0, df.columns.get_loc('EUR balance')] = eur_start_value
    eur_bal = 0
    # initializing backtest variables
    is_long = [False for i in buy_levels]
    buy_count = 0
    sell_count = 0
    long_trade_count = 0
    short_trade_count = 0
    max_eur_bal = 0
    max_eur_drawdown = 0
    current_eur_drawdown = 0
    max_usd_bal = 0
    max_usd_drawdown = 0
    current_usd_drawdown = 0
    # used to calculate average trade duration and winrate
    first_entry_time = False
    # the most recent position entry, if there was only one level, this will be the same
    last_entry_time = False
    # the timestamp of the last sell order used for both winrate and trade duration
    last_exit_time = False
    # used to calculate
    total_long_time = timedelta()
    total_short_time = timedelta()
    short_trade_win_count = 0
    long_trade_win_count = 0
    for minute in df.index:
        if True in is_long and df.loc[minute, 'sell signal']:
            df.loc[minute, 'sell price'] = df.loc[minute, 'close'] * (1 - FEE)
            is_long = [False for i in buy_levels]
            # balance sell logic
            usd_bal += eur_bal * df.loc[minute, 'sell price']
            usd_bal_before_any_eur_entries = usd_bal
            # all of the money is in usd now, so we write down this balance
            df.loc[minute, 'USD balance'] = usd_bal
            df.loc[minute, 'EUR balance'] = usd_bal / df.loc[minute, 'close']
            # always sell the entire balance
            eur_bal = 0
            sell_count += 1
            # if there was a buy already, calculate trade durations and win rates for both
            if first_entry_time:
                long_trade_count += 1
                total_long_time += minute - first_entry_time
                if df.loc[last_entry_time, 'avg buy price'] < df.loc[minute, 'sell price']:
                    long_trade_win_count += 1
                if last_exit_time:
                    short_trade_count += 1
                    total_short_time += first_entry_time - last_exit_time
                    if df.loc[last_exit_time, 'sell price'] > df.loc[last_entry_time, 'avg buy price']:
                        short_trade_win_count += 1
            last_exit_time = minute
        elif False in is_long:
            for buy_level in buy_levels:
                if not is_long[buy_level - 1] and df.loc[minute, f'buy signal{buy_level}']:
                    buy_lot = params['buy lots'][buy_level - 1] / 100
                    df.loc[minute, f'buy price{buy_level}'] = df.loc[minute, 'close'] * (1 + FEE)
                    is_long[buy_level - 1] = True
                    eur_bal += buy_lot * usd_bal_before_any_eur_entries / df.loc[minute, f'buy price{buy_level}']
                    usd_bal -= buy_lot * usd_bal_before_any_eur_entries
                    df.loc[minute, 'avg buy price'] = (usd_bal_before_any_eur_entries - usd_bal) / eur_bal
                    buy_count += 1
                    # saves timestamp for winrate calculations which are done when selling, since you never know which buy is last
                    last_entry_time = minute
                    # if this is our first buy after a sell
                    if buy_level == 1:
                        first_entry_time = minute
        # balance and drawdown logic
        if df.loc[minute, 'USD balance'] and df.loc[minute, 'EUR balance']:
            current_eur_balance = df.loc[minute, 'EUR balance']
            current_usd_balance = df.loc[minute, 'USD balance']
            if max_eur_bal != 0:
                current_eur_drawdown = ((max_eur_bal - current_eur_balance) / max_eur_bal) * 100
            if current_eur_balance > max_eur_bal:
                max_eur_bal = current_eur_balance
            elif current_eur_drawdown > max_eur_drawdown:
                max_eur_drawdown = current_eur_drawdown
            if max_usd_bal != 0:
                current_usd_drawdown = ((max_usd_bal - current_usd_balance) / max_usd_bal) * 100
            if current_usd_balance > max_usd_bal:
                max_usd_bal = current_usd_balance
            elif current_usd_drawdown > max_usd_drawdown:
                max_usd_drawdown = current_usd_drawdown
        win_rate = 0
        average_long_trade_duration = 0
        average_short_trade_duration = 0
        if long_trade_count and short_trade_count:
            average_long_trade_duration = total_long_time / long_trade_count
            average_short_trade_duration = total_short_time / short_trade_count
            win_rate = (long_trade_win_count + short_trade_win_count) / (short_trade_count + long_trade_count) * 100
        balance_value_eur = usd_bal / df["close"].iloc[-1] + eur_bal
        balance_value_usd = eur_bal * df["close"].iloc[-1] + usd_bal
        profit_usd = ((balance_value_usd - START_USD_BALANCE) / START_USD_BALANCE) * 100
        profit_eur = ((balance_value_eur - eur_start_value) / eur_start_value) * 100
        result['profit percent'] = {'EUR': profit_eur, 'USD': profit_usd}
        result['max drawdown'] = {'EUR': max_eur_drawdown, 'USD': max_usd_drawdown}
        result['trade count'] = {'buys': buy_count, 'sells': sell_count, 'long trades': long_trade_count, 'short trades': short_trade_count}
        result['win rate'] = {'overall': win_rate, 'long wins': long_trade_win_count, 'short wins': short_trade_win_count}
        result['avg duration'] = {'long trade': str(average_long_trade_duration), 'short trade': str(average_short_trade_duration)}
        backtest_total_time = datetime.now() - backtest_start_time
        result['backtest duration'] = str(backtest_total_time)
    return result

# takes in a list of params and returns results
#def get_results(procnum, return_dict, data, param_list, param_len):
def get_results(data, param_list):
    global finished_tests
    result_list = []
    for params in param_list:
        results = backtest(data, params)
        result_list.append(results)
        finished_tests += 1
        sys.stdout.write(f'\rProgress: {finished_tests}/{param_len}, {round(100 * finished_tests/param_len, 2)}%')

        path = f'unsortedresults/result{finished_tests-1}.json'
        with open(path, 'w') as result_file:
            json.dump(results, result_file, indent=4)
    #return_dict[procnum] = result_list
    return result_list

def get_results_mp(procnum, return_dict, data, param_list, param_len):
    result_list = []
    for params in param_list:
        results = backtest(data, params)
        result_list.append(results)
        new_count = return_dict['count'] + 1
        return_dict['count'] = new_count
        sys.stdout.write(f'\rProgress: {new_count}/{param_len}, {round(100 * new_count / param_len, 2)}%')

        path = f'unsortedresults/result{new_count - 1}.json'
        with open(path, 'w') as result_file:
            json.dump(results, result_file, indent=4)
    return_dict[procnum] = result_list

def main():
    global param_len
    start_time = datetime.now()
    data = add_indicators(populate_data_dict())
    df = data['EURUSDT']
    param_combinations = get_param_combinations()


    # param_combinations = [param_combinations[0], param_combinations[100], param_combinations[300], param_combinations[400]]

    # randomizing order of params for more even timing
    print(f'Start time: {start_time}')
    random.shuffle(param_combinations)

    #param_combinations = [param_combinations[0], param_combinations[23]]

    param_len = len(param_combinations)
    thread_count = MAX_THREADS -2
    tests_per_thread = len(param_combinations) / thread_count
    tests_per_thread = int(math.ceil(tests_per_thread))
    print(f'Running {tests_per_thread} backtests for each of {thread_count} processes\nTotal tests: {len(param_combinations)}')
    param_combination_chunks = [param_combinations[x:x+tests_per_thread] for x in range(0, len(param_combinations), tests_per_thread)]


    '''with ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_results, df, param_chunk) for param_chunk in param_combination_chunks]
    list_of_result_lists = [f.result() for f in futures]
    results = [result for result_list in list_of_result_lists for result in result_list]
    print(len(results))'''
    #results = get_results(df, param_combinations)

    # multiprocessing
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict['count'] = 0
    thread_list = []
    for i in range(len(param_combination_chunks)):
        thread_list.append(multiprocessing.Process(target=get_results_mp, args=(i, return_dict, df, param_combination_chunks[i], param_len)))
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()
    results = []
    for key in return_dict:
        if key != 'count':
            for result in return_dict[key]:
                results.append(result)

    # sorting results by profits
    results_sorted = []
    while len(results) > 0:
        best_profit = -100
        best_profit_index = np.nan
        for i in range(len(results)):
            #print(results[i])
            if results[i]['profit percent']['EUR'] >= best_profit:
                best_profit = results[i]['profit percent']['EUR']
                best_profit_index = i
        results_sorted.append(results[best_profit_index])
        del results[best_profit_index]
    results = results_sorted
    # outputs best performers
    for i in range(len(results)):
        path = f'results/result{i}.json'
        with open(path, 'w') as result_file:
            json.dump(results[i], result_file, indent=4)
    total_time = datetime.now() - start_time
    print(f'\nTotal time: {total_time}')

if __name__ == '__main__':
    main()
