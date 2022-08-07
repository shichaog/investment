import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import talib
import mplfinance as mpf
import numpy as np
import os
import akshare as ak

import time

# 查找字体路径
print(matplotlib.matplotlib_fname())
# 查找字体缓存路径
print(matplotlib.get_cachedir())

plt.rcParams['font.sans-serif'] = ['SimHei']# 显示中文
plt.rcParams['axes.unicode_minus'] = False		# 显示负号

#hs300_df=pd.read_csv('./沪深300_股票代码.csv', index_col=0)

#'科创50ETF':'sh588000',
#'食品饮料ETF':'sh515170',

view_list = {'300ETF', '500ETF'}

Stock_list = {'300ETF':'sh510300','500ETF':'sh510500'}

#for idx,row in hs300_df.iterrows():
for name,symbol in Stock_list.items():
    ticker_name = name
    filefullpath = './tmp/' + name + '.csv'
    if os.path.isfile(filefullpath):
        daily_df = pd.read_csv(filefullpath)
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
        daily_df.set_index("date", inplace=True)
    else:
        daily_df = ak.stock_zh_index_daily(symbol=symbol)
        #   stock_daily_df.index.name = 'date'
        daily_df['date'] = daily_df.index
        daily_df = daily_df.reset_index(drop=True)
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
        daily_df.set_index("date", inplace=True)
        daily_df.to_csv(filefullpath)

    daily_df = daily_df.sort_index(ascending=True)
    daily_df = daily_df.rename(columns={'open': 'Open', 'high': 'High', 'close': 'Close', 'low': 'Low', 'volume': 'Volume'})

    op = 'Open'
    cl = 'Close'
    hi = 'High'
    lo = 'Low'
    vo = 'Volume'

    #https://github.com/matplotlib/mplfinance/tree/master/tests
    MA_dist = dict(vlines=[daily_df.index[-20], daily_df.index[-60], daily_df.index[-200]], colors=['black', 'r', 'b'], linestyle='-.', linewidths=(1, 1, 1))
    daily_df.index = pd.DatetimeIndex(daily_df.index)
    daily_df.index.name = 'Date'

    daily_df['Year'] = daily_df.index.strftime('%Y')
    daily_df['Month'] = daily_df.index.strftime('%m')
    daily_df['Day'] = daily_df.index.strftime('%d')

    
    #calculate MA & EMA
    daily_df['EMA20'] = talib.EMA(daily_df[cl], timeperiod=20)
    daily_df['EMA60'] = talib.EMA(daily_df[cl], timeperiod=60)
    daily_df['EMA120'] = talib.EMA(daily_df[cl], timeperiod=120)

    
    daily_df['MA20'] = talib.MA(daily_df[cl], timeperiod=20)
    daily_df['MA60'] = talib.MA(daily_df[cl], timeperiod=60)
    daily_df['MA120'] = talib.MA(daily_df[cl], timeperiod=120)


    daily_df['BBANDS_upper'], daily_df['BBANDS_middle'], daily_df['BBANDS_lower'] = talib.BBANDS(
        daily_df[cl],
        timeperiod=20,
        # number of non-biased standard deviations from the mean
        nbdevup=2,
        nbdevdn=2,
        # Moving average type: simple moving average here
        matype=0)

    daily_df.fillna(0, inplace=True)

    daily_df['High_BBANDS_upper'] = daily_df["High"] - daily_df["BBANDS_upper"]
    daily_df['Low_BBANDS_lower'] = daily_df["Low"] - daily_df["BBANDS_lower"]

    tag_year = daily_df.iloc[0].at['Year']
    buy_day = []
    sell_day = []
    num_share = 0
    tot_vaule = 100
    year_profit = []

    buyed = False
    for count in range(0, len(daily_df)):
        year, month, day = daily_df.iloc[count]["Year"], daily_df.iloc[count]["Month"], daily_df.iloc[count]["Day"]
        if year == tag_year:
            if daily_df.iloc[count].at['High_BBANDS_upper'] > 0 and buyed is True:
                tot_vaule = num_share * daily_df.iloc[count].at["BBANDS_upper"]
                sell_day.append(year + "-" + month + "-" + day)
                buyed = False
            elif daily_df.iloc[count].at['Low_BBANDS_lower'] < 0 and buyed is False:
                num_share = int(tot_vaule / daily_df.iloc[count].at["BBANDS_lower"])
                buy_day.append(year + "-" + month + "-" + day)
                buyed = True
        else:
            year_profit.append(tag_year)
            year_profit.append(round((tot_vaule - 100) / 100, 4))
            num_share = 0
            tot_vaule = 100
            buyed = False
            tag_year = year
    print(year_profit)












    




