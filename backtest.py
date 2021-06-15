# plots and backtests eur price arbitrage against stable coins and usd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')

from datetime import datetime, timedelta
from binance.client import Client

# creates binance api client
client = Client('', '')

FOREX_DIR = 'mt_source_data'
FOREX_DIR_2 = 'mt_source_data_2'
DATA_DIR = 'all_price_data'

DF_COLUMNS = 'datetime open high low close volume'.split()
# numbered columns will be deleted
DF_BINANCE_COLUMNS = 'datetime open high low close 0 1 2 3 4 5 6'.split()

BASE_ASSETS = ['EUR', 'AUD', 'GBP']
FOREX_SYMBOLS = [f'{base}USD' for base in BASE_ASSETS]
STABLE_SYMBOLS = [f'{base}USDT' for base in BASE_ASSETS]

DAYMINS = 1440
EMA_LENGTHS = (DAYMINS * 3, DAYMINS * 1, DAYMINS * 7)

# backtest params
START_USD_BALANCE = 100
BUY_DIFF_P = -0.5
SELL_DIFF_P = -0.1
# binance fee when using BNB = 0.00075
FEE = 0.001

LINE = '-----------------------------------------------------------------------'

# gets df from csv
def get_forex_df(symbol):
    # df = pd.read_csv(f'{FOREX_DIR}/{symbol}.csv', index_col='date', parse_dates=True)
    df = pd.read_csv(f'{FOREX_DIR}/{symbol}.csv')
    df2 = pd.read_csv(f'{FOREX_DIR_2}/{symbol}.csv')
    df = pd.concat([df, df2], ignore_index=True, axis=0)
    # source data is in utc + 3
    df['datetime'] = pd.to_datetime(df['date'])
    df.set_index('datetime', inplace=True)
    df = df.tz_localize('Europe/Moscow')
    df = df.tz_convert("UTC")
    df.drop(['volume', 'date', 'Unnamed: 0'], axis=1, inplace=True)
    # removing rows with duplicate indexes
    df = df.loc[~df.index.duplicated(), :]
    return df

# gets stable coin data from binance
def get_stable_df(symbol, start, end):
    print(f'Downloading {symbol} data from binance')
    candles = client.get_historical_klines(symbol, '1m', start, end)
    df = pd.DataFrame(candles, dtype=float, columns=DF_BINANCE_COLUMNS)
    # make sure to put this data in the correct timezone
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df.tz_localize('UTC')
    # drops columns to match forex dataframe
    df.drop('0 1 2 3 4 5 6'.split(), axis=1, inplace=True)
    return df

# checks for data directories
def make_dirs():
    if not os.path.isdir(FOREX_DIR):
        os.mkdir(FOREX_DIR)
    if not os.path.isdir(FOREX_DIR_2):
        os.mkdir(FOREX_DIR_2)
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

# populates price data dict
def populate_data_dict(data):
    for forex_symbol in FOREX_SYMBOLS:
        file_path = f'{DATA_DIR}/{forex_symbol}.csv'
        if not os.path.isfile(file_path):
            data[forex_symbol] = get_forex_df(forex_symbol)
        else:
            data[forex_symbol] = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
            #data[forex_symbol] = data[forex_symbol].tz_localize('UTC')
    # trims all forex data to have a common start time and a common end time
    latest_start_time = False
    earliest_end_time = False
    for forex_symbol in FOREX_SYMBOLS:
        if latest_start_time:
            if latest_start_time < pd.to_datetime(data[forex_symbol].index[0]):
                latest_start_time = pd.to_datetime(data[forex_symbol].index[0])
        else:
            latest_start_time = pd.to_datetime(data[forex_symbol].index[0])
        if earliest_end_time:
            if earliest_end_time > pd.to_datetime(data[forex_symbol].index[-1]):
                earliest_end_time = pd.to_datetime(data[forex_symbol].index[-1])
        else:
            earliest_end_time = pd.to_datetime(data[forex_symbol].index[-1])
    for forex_symbol in FOREX_SYMBOLS:
        '''
        reached_first_datetime = False
        for minute in data[forex_symbol].index:
            if not reached_first_datetime:
                if minute < latest_start_time:
                    data[forex_symbol].drop(minute, inplace=True)
                else:
                    reached_first_datetime = True
        #inverse_df_index = data[forex_symbol].index[::-1]
        inverse_df = data[forex_symbol].iloc[::-1]
        exceeded_last_datetime = False
        for minute in inverse_df.index:
            if not exceeded_last_datetime:
                if minute > earliest_end_time:
                    data[forex_symbol].drop(minute, inplace=True)
                else:
                    exceeded_last_datetime = True
        data[forex_symbol] = inverse_df.iloc[::-1]
        '''
        data[forex_symbol] = data[forex_symbol][latest_start_time <= data[forex_symbol].index]
        data[forex_symbol] = data[forex_symbol][earliest_end_time >= data[forex_symbol].index]

    # getting the start time and end time from forex data into millisecond timestamps
    start_ts_ms = int(data[FOREX_SYMBOLS[0]].index[0].timestamp() * 1000)
    end_ts_ms = int(data[FOREX_SYMBOLS[0]].index[-1].timestamp() * 1000)
    # getting the prices from binance
    for stable_symbol in STABLE_SYMBOLS:
        file_path = f'{DATA_DIR}/{stable_symbol}.csv'
        # fetches stable coin price data if it is not yet present
        if not os.path.isfile(file_path):
            data[stable_symbol] = get_stable_df(stable_symbol, start_ts_ms, end_ts_ms)
            data[stable_symbol].to_csv(file_path)
        else:
            data[stable_symbol] = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
            #data[stable_symbol] = data[stable_symbol].tz_localize('UTC')
    # filling gaps in the forex data
    for forex_symbol in FOREX_SYMBOLS:
        last_price_minute = 0
        for minute in data[STABLE_SYMBOLS[0]].index:
            if minute in data[forex_symbol].index:
                last_price_minute = minute
            # if no price has been found yet, don't fill
            elif last_price_minute:
                data[forex_symbol].loc[minute] = data[forex_symbol].loc[last_price_minute]
        data[forex_symbol].sort_index(inplace=True)
    # saving forex csvs after all these changes are done
    for forex_symbol in FOREX_SYMBOLS:
        file_path = f'{DATA_DIR}/{forex_symbol}.csv'
        if not os.path.isfile(file_path):
            data[forex_symbol].to_csv(file_path)
    return data


# plots price and price difference
def plot_price_diff(data):
    # creates plot with 2 graphs
    fig = plt.figure()
    ax1 = fig.add_axes([0.07, 0.45, 0.9, 0.45])
    ax2 = fig.add_axes([0.07, 0.1, 0.9, 0.25])
    ax1.set_title('EUR 1m close price')
    # plots prices on top axis
    data['EURUSD'].plot(y='close', ax=ax1, label='USD', color='b', lw=0.5)
    data['EURUSDT'].plot(y='close', ax=ax1, label='USDT', color='g', lw=0.5)
    # data['EURBUSD'].plot(y='close', ax=ax1, label='BUSD', color='y', lw=0.5)
    # calculates difference in price
    # data['EURUSDT']['diff'] = data['EURUSDT']['close'] - data['EURUSD']['close']
    # data['EURBUSD']['diff'] = data['EURBUSD']['close'] - data['EURUSD']['close']
    # calculates percentage difference in prices
    data['EURUSDT']['diff'] = ((data['EURUSDT']['close'] - data['EURUSD']['close'])/data['EURUSD']['close']) * 100
    '''data['EURBUSD']['diff'] = ((data['EURBUSD']['close'] - data['EURUSD']['close'])/data['EURUSD']['close']) * 100
    # adding diff movin average and std
    data['EURUSDT']['EMA'] = data['EURUSDT']['diff'].ewm(span=EMA_LENGTHS[2], adjust=False).mean()
    data['EURUSDT']['LOWB'] = data['EURUSDT']['EMA'] - data['EURUSDT']['diff'].ewm(EMA_LENGTHS[2]).std()
    data['EURUSDT']['HIGHB'] = data['EURUSDT']['EMA'] + data['EURUSDT']['diff'].ewm(EMA_LENGTHS[2]).std()
    # plots difference on axis 2
    data['EURUSDT'].plot(y='diff', ax=ax2, label='€USDT-€USD', color='g', lw=0.5)
    data['EURBUSD'].plot(y='diff', ax=ax2, label='€BUSD-€USD', color='y', lw=0.5)
    # plotting mas
    data['EURUSDT'].plot(y='EMA', ax=ax2, label='EMA')
    data['EURUSDT'].plot(y='LOWB', ax=ax2, label='LOWB')
    data['EURUSDT'].plot(y='HIGHB', ax=ax2, label='HIGHB')'''
    plt.show()
    return data

# simulates trades on the data and writes them in new columns of the df
def populate_trades(data):
    for stable_symbol in STABLE_SYMBOLS:
        df = data[stable_symbol]
        df['buy_signal'] = df['diff'] < BUY_DIFF_P
        df['sell_signal'] = df['diff'] > SELL_DIFF_P
        df['buy_price'] = np.nan
        df['sell_price'] = np.nan
        is_long = False
        for minute in df.index:
            if not is_long:
                # checks for buy signal
                if df['buy_signal'].loc[minute]:
                    # writes buy price with fees included
                    df.loc[minute, 'buy_price'] = df['close'].loc[minute] * (1 + FEE)
                    is_long = True
            elif df['sell_signal'].loc[minute]:
                df.loc[minute, 'sell_price'] = df['close'].loc[minute] * (1 - FEE)
                is_long = False
        # this is probably uncessary but it feels wrong not doing it
        data[stable_symbol] = df
    return data

# plots trades
def plot_trades(data):
    for stable_symbol in STABLE_SYMBOLS:
        fig = plt.figure()
        ax = fig.add_axes([0.07, 0.1, 0.85, 0.8])
        ax.set_title(f'{stable_symbol} trades')
        # plots prices and trades
        data['EURUSD'].plot(y='close', ax=ax, label='EURUSD', color='b', lw=0.5)
        data[stable_symbol].plot(y='close', ax=ax, label=stable_symbol, color='y', lw=0.5)
        # adds another index column so scatterplots work for some unknown reason
        data[stable_symbol]['time'] = data[stable_symbol].index
        data[stable_symbol].plot.scatter(x='time', y='buy_price', ax=ax, marker='^', c='g', s=50, label='buy')
        data[stable_symbol].plot.scatter(x='time', y='sell_price', ax=ax, marker='v', c='r', s=50, label='sell')
        plt.show()

# back tests strategy
def backtest(data):
    for stable_symbol in STABLE_SYMBOLS:
        df = data[stable_symbol]
        usd_bal = START_USD_BALANCE
        eur_bal = 0
        eur_start = START_USD_BALANCE / df['close'].iloc[0]
        eur_start_str = '{:0.0{}f}'.format(eur_start, 2)
        print(f'\n\n{LINE}')
        print(f'{stable_symbol} backtest performance:\n{LINE}')
        print(f'Starting balance value:\nUSD: ${usd_bal}\nEUR: {eur_start_str}€\n{LINE}')
        is_long = False
        trade_count = 0
        long_trade_count = 0
        short_trade_count = 0
        # init an empty column
        df['EUR balance'] = np.nan
        df['USD balance'] = np.nan
        # plotting balance and calculating drawdown
        max_eur_bal = 0
        max_eur_drawdown = 0
        current_eur_drawdown = 0
        max_usd_bal = 0
        max_usd_drawdown = 0
        current_usd_drawdown = 0
        last_buy_minute = False
        last_sell_minute = False
        total_long_time = timedelta()
        total_short_time = timedelta()
        short_trade_win_count = 0
        long_trade_win_count = 0
        for minute in df.index:
            if not is_long:
                if df['buy_signal'].loc[minute]:
                    # buy
                    trade_count += 1
                    eur_bal = usd_bal / df['buy_price'].loc[minute]
                    usd_str = '{:0.0{}f}'.format(usd_bal, 2)
                    eur_str = '{:0.0{}f}'.format(eur_bal, 2)
                    print(f'{trade_count}. buy {eur_str}EUR for {usd_str}USD')
                    is_long = True
                    df.loc[minute, 'EUR balance'] = eur_bal
                    df.loc[minute, 'USD balance'] = usd_bal
                    usd_bal = 0
                    # calculates trade durations
                    last_buy_minute = minute
                    # if we have already sold before
                    if last_sell_minute:
                        short_trade_count += 1
                        total_short_time += minute - last_sell_minute
                        if df.loc[last_sell_minute, 'close'] > df.loc[minute, 'close']:
                            short_trade_win_count += 1
            else:
                if df['sell_signal'].loc[minute]:
                    # sell
                    trade_count += 1
                    usd_bal = eur_bal * df['sell_price'].loc[minute]
                    usd_str = '{:0.0{}f}'.format(usd_bal, 2)
                    eur_str = '{:0.0{}f}'.format(eur_bal, 2)
                    print(f'{trade_count}. sell {eur_str}EUR for {usd_str}USD')
                    eur_bal = 0
                    is_long = False
                    last_sell_minute = minute
                    # if there was a buy already
                    if last_buy_minute:
                        long_trade_count += 1
                        total_long_time += minute - last_buy_minute
                        if df.loc[last_buy_minute, 'close'] < df.loc[minute, 'close']:
                            long_trade_win_count += 1

            # balance graph and drawdown logic
            if df['EUR balance'].loc[minute]:
                current_eur_balance = df['EUR balance'].loc[minute]
                current_usd_balance = df.loc[minute, 'USD balance']
                if max_eur_bal != 0:
                    current_eur_drawdown = ((max_eur_bal-current_eur_balance)/max_eur_bal) * 100
                if current_eur_balance > max_eur_bal:
                    max_eur_bal = current_eur_balance
                elif current_eur_drawdown > max_eur_drawdown:
                    max_eur_drawdown = current_eur_drawdown

                if max_usd_bal != 0:
                    current_usd_drawdown = ((max_usd_bal-current_usd_balance)/max_usd_bal) * 100
                if current_usd_balance > max_usd_bal:
                    max_usd_bal = current_usd_balance
                elif current_usd_drawdown > max_usd_drawdown:
                    max_usd_drawdown = current_usd_drawdown

        average_long_trade_duration = total_long_time / long_trade_count
        average_short_trade_duration = total_short_time / short_trade_count
        print(LINE)
        print(f'Trade count (buys + sells): {trade_count}')
        print(f'Average long trade duration (buy -> sell): {average_long_trade_duration}')
        print(f'Long trade wins/losses: {long_trade_win_count}/{long_trade_count-long_trade_win_count} = {long_trade_win_count/long_trade_count*100}% winrate')
        print(f'Average short trade duration (sell -> buy): {average_short_trade_duration}')
        print(f'Short trade wins/losses: {short_trade_win_count}/{short_trade_count-short_trade_win_count} = {short_trade_win_count/short_trade_count * 100}% winrate')
        print(f'Overall winrate: {(long_trade_win_count+short_trade_win_count)/(short_trade_count+long_trade_count)*100}%')
        if is_long:
            usd_value = eur_bal * df["close"].iloc[-1]
            profit_usd = ((usd_value - START_USD_BALANCE)/START_USD_BALANCE) * 100
            profit_eur = ((eur_bal - eur_start)/eur_start) * 100
            print('Profit (USD%): {:0.0{}f}%'.format(profit_usd, 2))
            print('Profit (EUR%): {:0.0{}f}%'.format(profit_eur, 2))
            usd_str = '{:0.0{}f}'.format(usd_value, 2)
            eur_str = '{:0.0{}f}'.format(eur_bal, 2)
            print(f'Final balance value:\nUSD: ${usd_str}\nEUR: {eur_str}€')
        else:
            eur_value = usd_bal / df["close"].iloc[-1]
            profit_usd = ((usd_bal - START_USD_BALANCE)/START_USD_BALANCE) * 100
            profit_eur = ((eur_value - eur_start)/eur_start) * 100
            print('Profit (USD%): {:0.0{}f}%'.format(profit_usd, 2))
            print('Profit (EUR%): {:0.0{}f}%'.format(profit_eur, 2))
            usd_str = '{:0.0{}f}'.format(usd_bal, 2)
            eur_str = '{:0.0{}f}'.format(eur_value, 2)
            print(f'Final balance values:\nUSD: ${usd_str}\nEUR: {eur_str}€')
        print('Max drawdown (USD%): {:0.0{}f}%'.format(max_usd_drawdown, 2))
        print('Max drawdown (EUR%): {:0.0{}f}%'.format(max_eur_drawdown, 2))

        # plotting balance graph
        fig = plt.figure()
        ax = fig.add_axes([0.07, 0.1, 0.85, 0.8])
        ax.set_title(f'{stable_symbol} backtest balance graph in EUR and USD (not normalized %)')
        # plots prices and trades
        df[df['EUR balance'] > 0].plot(y='EUR balance', ax=ax, label='EUR', color='b')
        df[df['USD balance'] > 0].plot(y='USD balance', ax=ax, label='USD', color='g')
        plt.show()

def main():
    make_dirs()
    data = {}
    data = populate_data_dict(data)
    # this stuff is not updated for this script yet
    # data = plot_price_diff(data)
    # data = populate_trades(data)
    # plot_trades(data)
    # backtest(data)

if __name__ == '__main__':
    main()
'''import pandas
tz = pandas._libs.tslibs.strptime
print(tz)'''