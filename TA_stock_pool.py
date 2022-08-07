from matplotlib import font_manager as fm, rcParams
import matplotlib as plt

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
from calBuyPoint import *
from candle_rankings import *

import time

# 查找字体路径
print(matplotlib.matplotlib_fname())
# 查找字体缓存路径
print(matplotlib.get_cachedir())

plt.rcParams['font.sans-serif'] = ['simhei']# 显示中文
plt.rcParams['axes.unicode_minus'] = False		# 显示负号

#hs300_df=pd.read_csv('./沪深300_股票代码.csv', index_col=0)

#'科创50ETF':'sh588000',
#'食品饮料ETF':'sh515170',

view_list = {'300ETF', '500ETF', '光明乳业', '恒瑞医药'}


Stock_list = {
                '300ETF': 'sh510300', '500ETF': 'sh510500',
               # '中概互联ETF': 'sh513050',
          #      '标普500': 'sh513500', '纳指ETF': 'sh512100',
         #      '比亚迪': 'sz002594',
          #     '宁德时代': 'sz300750',
           #    '中国平安': 'sh601318',
            #   '恒瑞医药': 'sh600276', '通策医疗': 'sh600763', '药明康德': 'sh603259', '爱尔眼科':'sz300015', '安图生物': 'sh603658',
             #  '万科A': 'sz000002',
              # '片仔癀': 'sh600436', '云南白药': 'sz000538', '同仁堂': 'sh600085',
              # '海天味业': 'sh603288', '千禾味业': 'sh603027', '百润股份': 'sz002568',
              # '伊利股份': 'sh600887', '光明乳业': 'sh600597',
              # '美的集团': 'sz000333', '格力电器': 'sz000651',
              # '贵州茅台': 'sh600519', '五粮液': 'sz000858',
              # '招商银行': 'sh600036',
              # '中宠股份': 'sz002891', '比亚迪': 'sz002594',

}

#for idx,row in hs300_df.iterrows():
for ticker_name, symbol in Stock_list.items():
        filefullpath = './tmp/' + ticker_name + '.csv'
        if os.path.isfile(filefullpath):
            daily_df = pd.read_csv(filefullpath)
            daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
            daily_df.set_index("date", inplace=True)
        else:
            daily_df = ak.stock_zh_index_daily_tx(symbol=symbol)

            #   stock_daily_df.index.name = 'date'
            daily_df['date'] = daily_df.index
            daily_df = daily_df.reset_index(drop=True)
            daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
            daily_df.set_index("date", inplace=True)
            daily_df.to_csv(filefullpath)

        daily_df = daily_df.sort_index(ascending=True)
        daily_df = daily_df.rename(columns={'open': 'Open', 'high': 'High', 'close': 'Close', 'low': 'Low', 'amount': 'Volume' })

        if len(daily_df) > 500:
            daily_df = daily_df.tail(500)

        op = 'Open'
        cl = 'Close'
        hi = 'High'
        lo = 'Low'
        vo = 'Volume'

        #https://github.com/matplotlib/mplfinance/tree/master/tests
        if (len(daily_df) >=200):
            MA_dist = dict(vlines=[daily_df.index[-20], daily_df.index[-60], daily_df.index[-200]], colors=['black','r','b'], linestyle='-.', linewidths=(1,1,1))
        elif (len(daily_df) >=60):
            MA_dist = dict(vlines=[daily_df.index[-20], daily_df.index[-60]], colors=['black','r'], linestyle='-.', linewidths=(1,1))
        elif (len(daily_df) >=20):
            MA_dist = dict(vlines=[daily_df.index[-20]], colors=['black'], linestyle='-.', linewidths=(1))
        daily_df.index = pd.DatetimeIndex(daily_df.index)
        daily_df.index.name = 'Date'

        ohlc_dict = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
        month_df = daily_df.resample('M').agg(ohlc_dict)

        # calculate MA & EMA
        daily_df['EMA20'] = talib.EMA(daily_df[cl], timeperiod=20)
        daily_df['EMA60'] = talib.EMA(daily_df[cl], timeperiod=60)
        daily_df['EMA120'] = talib.EMA(daily_df[cl], timeperiod=120)
        EMA20_plot = mpf.make_addplot(daily_df["EMA20"], linestyle='dashed', color='black', width=0.5)
        EMA60_plot = mpf.make_addplot(daily_df["EMA60"], linestyle='dashed', color='r', width=0.5)
        EMA120_plot = mpf.make_addplot(daily_df["EMA120"], linestyle='dashed', color='b', width=0.5)

        daily_df['MA5'] = talib.MA(daily_df[cl], timeperiod=5)
        daily_df['MA20'] = talib.MA(daily_df[cl], timeperiod=20)
        daily_df['MA60'] = talib.MA(daily_df[cl], timeperiod=60)
        daily_df['MA120'] = talib.MA(daily_df[cl], timeperiod=120)
        MA5_plot = mpf.make_addplot(daily_df["MA5"], linestyle='solid', color='fuchsia', width=0.5)
        MA20_plot = mpf.make_addplot(daily_df["MA20"], linestyle='solid', color='black', width=0.5)
        MA60_plot = mpf.make_addplot(daily_df["MA60"], linestyle='solid', color='r', width=0.5)
        MA120_plot = mpf.make_addplot(daily_df["MA120"], linestyle='solid', color='b', width=0.5)

        daily_df["macd"], daily_df["macd_signal"], daily_df["macd_hist"] = talib.MACD(daily_df["Close"], fastperiod=12,slowperiod=26, signalperiod=9)

        daily_df.dropna()
        
        # macd panel
        colors = ['r' if v >= 0 else 'g' for v in daily_df["macd_hist"]]
        macd_plot = mpf.make_addplot(daily_df["macd"], panel=1, color='fuchsia', title="MACD", width=0.5)
        macd_hist_plot = mpf.make_addplot(daily_df["macd_hist"], type='bar', panel=1, color=colors, width=0.5) # color='dimgray'
        macd_signal_plot = mpf.make_addplot(daily_df["macd_signal"], panel=1, color='b', width=0.5)

        #K line pattern
        daily_df = find_candle_pattern(daily_df)

        # angle 趋势线角30 45度，45度趋势最强，70 80度趋势不稳定
        daily_df['EMA20_angle'] = round(talib.LINEARREG_ANGLE(daily_df['EMA20'], timeperiod=14),1)
        daily_df['EMA60_angle'] = round(talib.LINEARREG_ANGLE(daily_df['EMA60'], timeperiod=14),1)
        daily_df['EMA120_angle'] = round(talib.LINEARREG_ANGLE(daily_df['EMA120'], timeperiod=14),1)
        #Trading only happed when MA20 line above MA60 line
        #when position equals 1, meas MA20 up cross MA60/ close price up cross EMA20
        daily_df['MA_cross'] = np.where(daily_df['MA20'] > daily_df['MA60'], 1.0, 0.0)
        daily_df['MA_cross'] = daily_df['MA_cross'].diff()
        daily_df['EMA_price_cross'] = np.where(daily_df['Close'] > daily_df['EMA20'], 1.0, 0.0)
        daily_df['EMA_price_cross'] = daily_df['EMA_price_cross'].diff()
        #MACD cross over
        daily_df["macd_cross"] = np.where(daily_df["macd"] > daily_df["macd_signal"], 1.0, 0.0)
        daily_df['macd_cross'] = daily_df['macd_cross'].diff()
        
        daily_df['2cross_1'] = daily_df["macd_cross"] + daily_df['EMA_price_cross'] + daily_df['EMA_price_cross'].shift(1) + daily_df['EMA_price_cross'].shift(2) + daily_df['EMA_price_cross'].shift(3) + daily_df['EMA_price_cross'].shift(4) + daily_df['EMA_price_cross'].shift(5)  + daily_df['EMA_price_cross'].shift(6)
        daily_df['2cross_2'] = daily_df['EMA_price_cross'] + daily_df["macd_cross"] + daily_df["macd_cross"].shift(1) + daily_df["macd_cross"].shift(2) + daily_df["macd_cross"].shift(3) + daily_df["macd_cross"].shift(4) + daily_df["macd_cross"].shift(5)  + daily_df["macd_cross"].shift(6)
        #2 cross        
        daily_df['buy_marker'] = np.where((daily_df['2cross_1'] >= 2.0) | (daily_df['2cross_2'] >= 2.0), daily_df['Low'],np.nan)
        daily_df['sell_marker'] = np.where((daily_df['2cross_1'] <= -2.0) | (daily_df['2cross_2'] <= -2.0), daily_df['High'],np.nan)
        up_plot = mpf.make_addplot(daily_df['buy_marker'], type='scatter', marker='^', markersize=1.5, panel=0, color='r')
        down_plot = mpf.make_addplot(daily_df['sell_marker'], type='scatter', marker='v', markersize=1.5, panel=0, color='b')
        if daily_df.iloc[-1]["buy_marker"] == daily_df.iloc[-1]["Low"]:
            print("Buy ", ticker_name)
        if daily_df.iloc[-1]["sell_marker"] == daily_df.iloc[-1]["High"]:
            print("Sell ", ticker_name)

        if daily_df.iloc[-1]['buy_marker'] == daily_df.iloc[-1]['Low']:
            print(ticker_name + ' buy point')

        if daily_df.iloc[-1]['sell_marker'] == daily_df.iloc[-1]['High']:
            print(ticker_name + ' sell point')

        plots = [up_plot, down_plot, MA5_plot, MA20_plot, MA60_plot, MA120_plot, EMA20_plot, EMA60_plot, EMA120_plot, macd_plot, macd_signal_plot, macd_hist_plot]
    
        #设置k线图颜色
        my_color = mpf.make_marketcolors(up='red',#上涨时为红色
                                     down='green',#下跌时为绿色
                                     edge='i',#隐藏k线边缘
                                     volume='in',#成交量用同样的颜色
                                     inherit=True)
        my_style = mpf.make_mpf_style(gridaxis='both',#设置网格
                               gridstyle='-.',
                               y_on_right=True,
                                marketcolors=my_color)


        fig, axlist = mpf.plot(daily_df,
             type='candle',
             style=my_style,
             addplot=plots,
             title=f"\n{ticker_name}",
             volume=True,
             vlines=MA_dist,
             volume_panel=2,
             ylabel='',
             ylabel_lower='',
             figratio=(2,1),
             returnfig=True)
            # savefig='test-mplfiance.png')




         
        axlist[0].text(len(daily_df), daily_df.iloc[-1]['EMA20'], str(daily_df.iloc[-1]['EMA20_angle']), fontsize=1.8)
        axlist[0].text(len(daily_df), daily_df.iloc[-1]['EMA60'], str(daily_df.iloc[-1]['EMA60_angle']), fontsize=1.8)
        axlist[0].text(len(daily_df), daily_df.iloc[-1]['EMA120'], str(daily_df.iloc[-1]['EMA120_angle']), fontsize=1.8)

        count = 0
        for index, row in daily_df.iterrows():
            count += 1
            if daily_df.loc[index, 'candlestick_match_count'] >= 1:
                axlist[0].text(count - 1, daily_df.loc[index, 'High'] * 1.02,
                                                (daily_df.loc[index, 'candlestick_pattern'])[3:], fontsize=1.8)


        fig.savefig( ticker_name + daily_df.index[-1].strftime('%Y-%m-%d') + '.png', dpi=500)





