#!/usr/bin/env python
#coding=utf-8

import pandas as pd
from pandas import Series
import numpy as np
import talib
from candle_rankings import *
import math

def TrendLine(daily_df):
    DRAW = False
    support_tupple = []
    resistance_tupple = []
    df = daily_df.reset_index()
    # Chose local maximum
    support_df = df.loc[
        (df['Close'] > df['Close'].shift(1))
        & (df['Close'] > df['Close'].shift(2))
        & (df['Close'] > df['Close'].shift(-1))
        & (df['Close'] > df['Close'].shift(-2))
    ]
    # Chose highest value of five days and no 8% difference
    idx = 0
    while idx < (len(support_df)-1):
        if ((support_df.index[idx+1] - support_df.index[idx]) <= 5):
            if support_df.loc[support_df.index[idx+1], 'Close'] > support_df.loc[support_df.index[idx], 'Close'] and support_df.loc[support_df.index[idx+1], 'Close'] < support_df.loc[support_df.index[idx], 'Close']* 1.08:
                support_df.drop(support_df.index[idx], axis=0, inplace=True)
                idx = idx-1
            elif support_df.loc[support_df.index[idx+1], 'Close'] * 1.08 < support_df.loc[support_df.index[idx], 'Close']:
                support_df.drop(support_df.index[idx+1], axis=0, inplace=True)
                i = idx-1
        idx = idx + 1

    idx = 0
    while idx < (len(support_df)-2):
        #求向量夹角
        #   /a
        #b
        #  \c
        a_x, b_x, c_x = support_df.index[idx+1], support_df.index[idx+2], support_df.index[idx]
        a_y, b_y, c_y = support_df.loc[support_df.index[idx+1], 'Close'], support_df.loc[support_df.index[idx+2],'Close'], support_df.loc[support_df.index[idx], 'Close']
        x1, y1 = (a_x - b_x), (a_y - b_y)
        x2, y2 = (c_x - b_x), (c_y - b_y)
        cos_b = (x1*x2 + y1*y2) / ((math.sqrt(x1**2 + y1**2))*math.sqrt(x2**2 + y2**2))
        angle = math.degrees(math.acos(cos_b))
        angle_dis = angle * (support_df.index[idx+2] - support_df.index[idx+1])
        if (math.fabs(angle) * (support_df.index[idx+2] - support_df.index[idx+1]) < 100):
            if (support_df.loc[support_df.index[idx+1], 'Volume'] >  support_df.loc[support_df.index[idx+2], 'Volume'] * 1.3 ):
                support_df.drop(support_df.index[idx + 2], axis=0, inplace=True)
            elif (support_df.loc[support_df.index[idx+2], 'Volume'] >  support_df.loc[support_df.index[idx+1], 'Volume'] * 1.3):
                support_df.drop(support_df.index[idx + 1], axis=0, inplace=True)
            elif support_df.loc[support_df.index[idx+2], 'Close'] > support_df.loc[support_df.index[idx+1], 'Close']:
                support_df.drop(support_df.index[idx + 1], axis=0, inplace=True)
            else:
                support_df.drop(support_df.index[idx + 2], axis=0, inplace=True)
            idx = idx - 1
        idx = idx + 1

    resistance_df = df.loc[
        (df['Close'] < df['Close'].shift(1))
        & (df['Close'] < df['Close'].shift(2))
        & (df['Close'] < df['Close'].shift(-1))
        & (df['Close'] < df['Close'].shift(-2))
    ]

    idx = 0
    while idx < (len(resistance_df)-1):
        if ((resistance_df.index[idx+1] - resistance_df.index[idx]) <= 5):
            if resistance_df.loc[resistance_df.index[idx+1], 'Close'] < resistance_df.loc[resistance_df.index[idx], 'Close'] and resistance_df.loc[resistance_df.index[idx+1], 'Close'] < resistance_df.loc[resistance_df.index[idx], 'Close']* 1.08:
                resistance_df.drop(resistance_df.index[idx], axis=0, inplace=True)
                idx = idx-1
            elif resistance_df.loc[resistance_df.index[idx+1], 'Close'] * 1.08 > resistance_df.loc[resistance_df.index[idx], 'Close']:
                resistance_df.drop(resistance_df.index[idx+1], axis=0, inplace=True)
                i = idx-1
        idx = idx + 1

    idx = 0
    while idx < (len(resistance_df)-2):
        #求向量夹角
        #   /a
        #b
        #  \c
        a_x, b_x, c_x = resistance_df.index[idx+1], resistance_df.index[idx+2], resistance_df.index[idx]
        a_y, b_y, c_y = resistance_df.loc[resistance_df.index[idx+1], 'Close'], resistance_df.loc[resistance_df.index[idx+2],'Close'], resistance_df.loc[resistance_df.index[idx], 'Close']
        x1, y1 = (a_x - b_x), (a_y - b_y)
        x2, y2 = (c_x - b_x), (c_y - b_y)
        cos_b = (x1*x2 + y1*y2) / ((math.sqrt(x1**2 + y1**2))*math.sqrt(x2**2 + y2**2))
        print(cos_b)
        if cos_b >= 1:
            cos_b = cos_b - 0.00001
        angle = math.degrees(math.acos(cos_b))
        angle_dis = angle * (resistance_df.index[idx+2] - resistance_df.index[idx+1])
        if (math.fabs(angle) * (resistance_df.index[idx+2] - resistance_df.index[idx+1]) < 100):
            if (resistance_df.loc[resistance_df.index[idx+1], 'Volume'] >  resistance_df.loc[resistance_df.index[idx+2], 'Volume'] * 1.3 ):
                resistance_df.drop(resistance_df.index[idx + 2], axis=0, inplace=True)
            elif (resistance_df.loc[resistance_df.index[idx+2], 'Volume'] >  resistance_df.loc[resistance_df.index[idx+1], 'Volume'] * 1.3):
                resistance_df.drop(resistance_df.index[idx + 1], axis=0, inplace=True)
            elif resistance_df.loc[resistance_df.index[idx+2], 'Close'] < resistance_df.loc[resistance_df.index[idx+1], 'Close']:
                resistance_df.drop(resistance_df.index[idx + 1], axis=0, inplace=True)
            else:
                resistance_df.drop(resistance_df.index[idx + 2], axis=0, inplace=True)
            idx = idx - 1
        idx = idx + 1

    angle_support = math.degrees(math.atan((support_df.loc[support_df.index[-1], 'Close'] - support_df.loc[support_df.index[-2], 'Close'])/(support_df.index[-1] -support_df.index[-2])))
    angle_resistance = math.degrees(math.atan((resistance_df.loc[resistance_df.index[-1], 'Close'] - resistance_df.loc[resistance_df.index[-2], 'Close']) / (
                resistance_df.index[-1] - resistance_df.index[-2])))

    if (angle_support < 5 and angle_support >-5):
        DRAW = True

        if ((len(support_df) >= 3) and (len(resistance_df) >=3)):
            support_df.set_index('Date', inplace=True)
            support_tupple.append((support_df.index[-1], support_df.loc[support_df.index[-1], 'Close']))
            support_tupple.append((support_df.index[-2], support_df.loc[support_df.index[-2], 'Close']))
            support_tupple.append((support_df.index[-3], support_df.loc[support_df.index[-3], 'Close']))
            resistance_df.set_index('Date', inplace=True)
            resistance_tupple.append((resistance_df.index[-1], resistance_df.loc[resistance_df.index[-1], 'Close']))
            resistance_tupple.append((resistance_df.index[-2], resistance_df.loc[resistance_df.index[-2], 'Close']))
            resistance_tupple.append((resistance_df.index[-3], resistance_df.loc[resistance_df.index[-3], 'Close']))
        elif ((len(support_df) >= 1) and (len(resistance_df) >= 1)):
            support_df.set_index('Date', inplace=True)
            support_tupple.append((support_df.index[-1], support_df.loc[support_df.index[-1], 'Close']))
            resistance_df.set_index('Date', inplace=True)
            resistance_tupple.append((resistance_df.index[-1], resistance_df.loc[resistance_df.index[-1], 'Close']))
        else:
            support_tupple = []
            resistance_tupple = []

    return DRAW, support_tupple, resistance_tupple



#均线买点策略
def FindMABuyPoint(df):
    buy_date = []
    cnt = 2
    df['MA_crossover'] = np.where(df['Close'] > df['MA20'], 1.0, 0.0)
    df['MA_signal'] = df['MA_crossover'].diff()
    df['MACD_crossover'] = np.where(df['macd'] > df["macd_signal"], 1.0, 0.0)
    df['MACD_siganl'] = df['MACD_crossover'].diff()
    df['KD_crossover'] = np.where(df['slowk'] > df["slowd"], 1.0, 0.0)
    df['KD_siganl'] = df['KD_crossover'].diff()



    buy_signal = df.loc[
        #MACD cross before price cross
        (
            # Rule 1: price upper cross MA10
            (df['Close'] > df['MA20']) & (df['Close'].shift(5) < df['MA20'].shift(5))
            # Rule 2: MACD upper cross
            & (df['macd'] > df["macd_signal"]) & (df['macd'].shift(5) < df["macd_signal"].shift(5))
            #Rule 3: EMA5 upper cross SMA15
            & (df['EMA20'] > df['EMA60']) & (df['EMA60'] > df['EMA120'])
        )
        #Price cross before MACD cross
    ]

    return buy_signal.index.tolist()
        
##均线卖点策略
def FindMASellPoint(df):
    df['50'] = 50
    sell_signal = df.loc[
        #MACD cross before price cross
        (
            # Rule 1: price upper cross MA10
            (df['Close'] < df['MA20']*1.1) & (df['Close'].shift(5) > df['MA20'].shift(5))
            # Rule 2: MACD upper cross
            & (df['macd'] < df["macd_signal"]) & (df['macd'].shift(5) > df["macd_signal"].shift(5))
            #Rule 3: EMA5 upper cross SMA15
            & (df['EMA20'] < df['MA20']) & (df['EMA20'].shift(5) > df['EMA120'].shift(5))
        )

        #Price cross before MACD cross
    ]
    return sell_signal.index.tolist()

def calBuySellPoint(df, price_flat_percent=3, volum_increase_percent=50):
    MA_buy_date = FindMABuyPoint(df)
    MA_sell_date = FindMASellPoint(df)


    return MA_buy_date, MA_sell_date

    
    

