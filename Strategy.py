#!/usr/bin/env python
#coding=utf-8

import pandas as pd
from pandas import Series
import numpy as np
import talib
from candle_rankings import *


def Strategy(df, price_flat_percent=3, volum_increase_percent=50):
    MA_buy_date = FindMABuyPoint(df)
    MA_sell_date = FindMASellPoint(df)

  #  for idx_date in MA_buy_date:
   #     if isVolLarge(df, idx_date, percent=25) == False:

    vol_buy_date = isVolBuyPoint(df, price_vary_rate=10, vol_change_rate=50)
    vol_sell_date = isVolSellPoint(df, price_vary_rate=10, vol_change_rate=50)

    return MA_buy_date, MA_sell_date
