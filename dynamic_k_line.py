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
import matplotlib.animation as animation

import time

# 查找字体路径
print(matplotlib.matplotlib_fname())
# 查找字体缓存路径
print(matplotlib.get_cachedir())

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

view_list = {'300ETF', '500ETF'}

Stock_list = {'300ETF':'sh510300'}
# Stock_list = {'300ETF':'sh510300','500ETF':'sh510500'}

# for idx,row in hs300_df.iterrows():
for name, symbol in Stock_list.items():
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
    daily_df = daily_df.rename(
        columns={'open': 'Open', 'high': 'High', 'close': 'Close', 'low': 'Low', 'volume': 'Volume'})

    op = 'Open'
    cl = 'Close'
    hi = 'High'
    lo = 'Low'
    vo = 'Volume'

    # https://github.com/matplotlib/mplfinance/tree/master/tests
    daily_df.index = pd.DatetimeIndex(daily_df.index)
    daily_df.index.name = 'Date'

    # 设置k线图颜色
    my_color = mpf.make_marketcolors(up='red',  # 上涨时为红色
                                     down='green',  # 下跌时为绿色
                                     edge='i',  # 隐藏k线边缘
                                     volume='in',  # 成交量用同样的颜色
                                     inherit=True)
    my_style = mpf.make_mpf_style(gridaxis='both',  # 设置网格
                                  gridstyle='-.',
                                  y_on_right=True,
                                  marketcolors=my_color)

    # calculate MA & EMA
    daily_df['EMA20'] = talib.EMA(daily_df[cl], timeperiod=20)
    daily_df['EMA60'] = talib.EMA(daily_df[cl], timeperiod=60)
    daily_df['EMA120'] = talib.EMA(daily_df[cl], timeperiod=120)
    EMA20_plot = mpf.make_addplot(daily_df["EMA20"], linestyle='dashed', color='black', width=0.5)
    EMA60_plot = mpf.make_addplot(daily_df["EMA60"], linestyle='dashed', color='r', width=0.5)
    EMA120_plot = mpf.make_addplot(daily_df["EMA120"], linestyle='dashed', color='b', width=0.5)

    daily_df['MA20'] = talib.MA(daily_df[cl], timeperiod=20)
    daily_df['MA60'] = talib.MA(daily_df[cl], timeperiod=60)
    daily_df['MA120'] = talib.MA(daily_df[cl], timeperiod=120)
    MA20_plot = mpf.make_addplot(daily_df["MA20"], linestyle='solid', color='black', width=0.5)
    MA60_plot = mpf.make_addplot(daily_df["MA60"], linestyle='solid', color='r', width=0.5)
    MA120_plot = mpf.make_addplot(daily_df["MA120"], linestyle='solid', color='b', width=0.5)

    daily_df["macd"], daily_df["macd_signal"], daily_df["macd_hist"] = talib.MACD(daily_df["Close"]);

    # macd panel
    colors = ['g' if v >= 0 else 'r' for v in daily_df["macd_hist"]]
    macd_plot = mpf.make_addplot(daily_df["macd"], panel=1, color='fuchsia', title="MACD")
    macd_hist_plot = mpf.make_addplot(daily_df['macd_hist'], type='bar', panel=1, color=colors)  # color='dimgray'
    macd_signal_plot = mpf.make_addplot(daily_df["macd_signal"], panel=1, color='b')

    add = [MA20_plot, MA60_plot, MA120_plot, EMA20_plot, EMA60_plot, EMA120_plot, macd_plot, macd_hist_plot,
           macd_signal_plot]

    kwargs = dict(type='candle', volume=True, volume_panel=2, style=my_style, addplot=add)
    # fig, axes = mpf.plot(daily_df, returnfig=True, **kwargs)

    warmup = 240
    _EMA20_plot = mpf.make_addplot(daily_df.iloc[0:warmup]["EMA20"], linestyle='dashed', color='black', width=0.5)
    _EMA60_plot = mpf.make_addplot(daily_df.iloc[0:warmup]["EMA60"], linestyle='dashed', color='r', width=0.5)
    _EMA120_plot = mpf.make_addplot(daily_df.iloc[0:warmup]["EMA120"], linestyle='dashed', color='b', width=0.5)
    _MA20_plot = mpf.make_addplot(daily_df.iloc[0:warmup]["MA20"], linestyle='solid', color='black', width=0.5)
    _MA60_plot = mpf.make_addplot(daily_df.iloc[0:warmup]["MA60"], linestyle='solid', color='r', width=0.5)
    _MA120_plot = mpf.make_addplot(daily_df.iloc[0:warmup]["MA120"], linestyle='solid', color='b', width=0.5)

    colors = ['g' if v >= 0 else 'r' for v in daily_df["macd_hist"]]
    _macd_plot = mpf.make_addplot(daily_df[0:warmup]["macd"], panel=1, color='fuchsia', title="MACD")
    _macd_hist_plot = mpf.make_addplot(daily_df[0:warmup]['macd_hist'], type='bar', panel=1,
                                       color=colors)  # color='dimgray'
    _macd_signal_plot = mpf.make_addplot(daily_df[0:warmup]["macd_signal"], panel=1, color='b')

    _add_plots = [_MA20_plot, _MA60_plot, _MA120_plot, _EMA20_plot, _EMA60_plot, _EMA120_plot, _macd_plot,
                  _macd_hist_plot, _macd_signal_plot]

    _kwargs = dict(type='candle', volume=True, volume_panel=2, style=my_style, addplot=_add_plots, title=name,
                   figsize=(16, 9))
    fig, axlist = mpf.plot(daily_df.iloc[0:warmup], returnfig=True, **_kwargs)

    ax_main = axlist[0]
    ax_emav = ax_main
    ax_hisg = axlist[2]
    ax_macd = axlist[3]
    ax_sign = ax_macd
    ax_volu = axlist[4]

    def run_animation():
        anim_running = True

        def onClick(event):
            nonlocal anim_running
            if anim_running:
                ani.event_source.stop()
                anim_running = False
            else:
                ani.event_source.start()
                anim_running = True

        def animate(i):
            if (20 + i) > len(daily_df):
                print('no more data to plot')
                ani.event_source.interval *= 3
                if ani.event_source.interval > 12000:
                    exit()
                return
            _daily_df = daily_df.iloc[i:(warmup + i)]
            _EMA20_plot = mpf.make_addplot(_daily_df["EMA20"], linestyle='dashed', color='black', width=0.5, ax=ax_emav)
            _EMA60_plot = mpf.make_addplot(_daily_df["EMA60"], linestyle='dashed', color='r', width=0.5, ax=ax_emav)
            _EMA120_plot = mpf.make_addplot(_daily_df["EMA120"], linestyle='dashed', color='b', width=0.5, ax=ax_emav)
            _MA20_plot = mpf.make_addplot(_daily_df["MA20"], linestyle='solid', color='black', width=0.5, ax=ax_emav)
            _MA60_plot = mpf.make_addplot(_daily_df["MA60"], linestyle='solid', color='r', width=0.5, ax=ax_emav)
            _MA120_plot = mpf.make_addplot(_daily_df["MA120"], linestyle='solid', color='b', width=0.5, ax=ax_emav)
            colors = ['g' if v >= 0 else 'r' for v in daily_df["macd_hist"]]
            _macd_plot = mpf.make_addplot(_daily_df["macd"], panel=1, color='fuchsia', title="MACD", ax=ax_macd)
            _macd_hist_plot = mpf.make_addplot(_daily_df['macd_hist'], type='bar', panel=1,
                                               color=colors, ax=ax_hisg)  # color='dimgray'
            _macd_signal_plot = mpf.make_addplot(_daily_df["macd_signal"], panel=1, color='b', ax=ax_sign)
            _add_plots = [_MA20_plot, _MA60_plot, _MA120_plot, _EMA20_plot, _EMA60_plot, _EMA120_plot, _macd_plot,
                          _macd_hist_plot, _macd_signal_plot]
            MA_dist = dict(vlines=[_daily_df.index[-20], _daily_df.index[-60], _daily_df.index[-200]],
                           colors=['black', 'r', 'b'], linestyle='-.', linewidths=(1, 1, 1))
            for ax in axlist:
                ax.clear()
            _kwargs = dict(type='candle', style=my_style, vlines=MA_dist, addplot=_add_plots, figsize=(16, 9))
            mpf.plot(_daily_df, ax=ax_main, volume=ax_volu, volume_panel=2, **_kwargs)

        fig.canvas.mpl_connect('button_press_event', onClick)
        ani = animation.FuncAnimation(fig, animate, interval=1000)

        #brew install imagemagick
        ani.save('stock.gif', writer='imagemagick', fps=25)


run_animation()
mpf.show()
