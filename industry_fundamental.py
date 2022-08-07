import pandas as pd
import os
import akshare as ak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.dates as mdates
import talib
from sklearn import preprocessing

my_year_month_day_fmt = mdates.DateFormatter('%Y-%m-%d')

DATA_LENGTH=int(180)
AVG_LENGTH=int(20)
#5 year PE
YEAR_PE=int(1220)

pe_pb_index = ['000300.XSHG', '000905.XSHG']
pe_pb300 =[]
pe_pb500 = []

for name in pe_pb_index:
    # 市盈率百分位
    stock_a_pe_list = ak.stock_a_pe(market=name)['middlePETTM'].tail(YEAR_PE).tolist()
    curr_pe = stock_a_pe_list[-1]
    stock_a_pe_list = sorted(stock_a_pe_list)
    index = stock_a_pe_list.index(curr_pe)
    pe_percent = f"{((index + 1) / YEAR_PE) * 100:.1f}"

    # 市净率百分位
    stock_a_pb_list = ak.stock_a_pb(market=name)['middlePB'].tail(YEAR_PE).tolist()
    curr_pb = stock_a_pb_list[-1]
    stock_a_pe_list = sorted(stock_a_pb_list)
    index = stock_a_pb_list.index(curr_pb)
    pb_percent = f"{((index + 1) / YEAR_PE) * 100:.1f}"
    if name == '000300.XSHG':
        pe_pb300.append(curr_pe)
        pe_pb300.append(pe_percent)
        pe_pb300.append(curr_pb)
        pe_pb300.append(pb_percent)
    elif name == '000905.XSHG':
        pe_pb500.append(curr_pe)
        pe_pb500.append(pe_percent)
        pe_pb500.append(curr_pb)
        pe_pb500.append(pb_percent)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(32, 18),
                                            gridspec_kw={'height_ratios': [ 1,2, 3.5]})

#Plot 沪深800指数数据
csrc300=['沪深300', '中证500']
#csrc300=['沪深300']
stock_zh_index_spot_df = ak.stock_zh_index_spot()

for name in csrc300:
    stock_index_df = stock_zh_index_spot_df.loc[stock_zh_index_spot_df['名称'] == name]
    filefullpath = './tmp/' + name + '.csv'
    if os.path.isfile(filefullpath):
        industry_daily_df = pd.read_csv(filefullpath)
        industry_daily_df = industry_daily_df.tail(DATA_LENGTH)
        industry_daily_df["date"] = pd.to_datetime(industry_daily_df["date"]).dt.date
        industry_daily_df.set_index("date", inplace=True)
    else:
        industry_daily_df = ak.stock_zh_index_daily(symbol=stock_index_df.iloc[0, 0])
        #   stock_daily_df.index.name = 'date'
        industry_daily_df['date'] = industry_daily_df.index
        industry_daily_df = industry_daily_df.reset_index(drop=True)
        industry_daily_df["date"] = pd.to_datetime(industry_daily_df["date"]).dt.date
        industry_daily_df.set_index("date", inplace=True)
        industry_daily_df.to_csv(filefullpath)
        industry_daily_df = industry_daily_df.tail(DATA_LENGTH)

    # industry_daily_df["close"] = industry_daily_df["close"].apply(np.log)
    industry_daily_df['close'] = (industry_daily_df['close'] - industry_daily_df['close'].mean()) / industry_daily_df['close'].mean()
    ax1.plot(industry_daily_df['close'], label='归一化'+name)

ax1.set_title('行业宽度'+industry_daily_df.index[-1].strftime('%Y-%m-%d') +' @ shichaog')


if(float(pe_pb300[1]) > 50 and float(pe_pb500[1]) > 50):
    color = 'r'
else:
    color = 'g'
ax1.text(industry_daily_df.index[-30], 0.25, '沪深300PE：'+ str(pe_pb300[0]) + ' 5年PE百分位：' + str(pe_pb300[1]) + ' PB：'+ str(pe_pb300[2]) + ' PB百分位：' + str(pe_pb300[3]) +
              '\n中证500PE：' + str(pe_pb500[0]) + ' 5年PE百分位：' + str(pe_pb500[1]) + ' PB：' + str(pe_pb500[2]) + ' PB百分位：' + str(pe_pb500[3]), fontsize=12, color = color)
ax1.legend()
ax1.set_xlim(industry_daily_df.index[0], industry_daily_df.index[-1])

#get industry class
#wget http://www.swsindex.com/idx0530.aspx
stock_indust_df = pd.read_excel('./SwClass.xls')

for idx, row in stock_indust_df.iterrows():
    code_len = len(str(row['股票代码']))
    new_code = str(row['股票代码'])
    if code_len < 6:
        while code_len < 6:
            new_code = '0' + new_code
            code_len = code_len + 1
        stock_indust_df.loc[idx, '股票代码'] = new_code
    else:
        stock_indust_df.loc[idx, '股票代码'] = str(stock_indust_df.loc[idx, '股票代码'])

fund_index = ['000905', '000300']
for name in fund_index:
    fund_df = ak.index_stock_cons(index=name)

    for idx, row in fund_df.iterrows():
        index = stock_indust_df[stock_indust_df["股票代码"] == row['品种代码']].index.tolist()[0]
        fund_df.loc[idx, '申万分类'] = stock_indust_df.loc[index, '行业名称']

    grouped_industry = fund_df.groupby('申万分类').size().sort_values(ascending=False)

    #替换行业分类中公司少的
    industry_list = list(grouped_industry.index)[-15:]
    fund_df['申万分类'] = fund_df['申万分类'].replace(industry_list, 'mixed')

    grouped_industry = fund_df.groupby('申万分类').size().sort_values(ascending=False)
    industry_list = list(grouped_industry.index)

    industry_df = pd.DataFrame(np.zeros(DATA_LENGTH * len(industry_list)).reshape(DATA_LENGTH, len(industry_list)),columns=industry_list, dtype='int32').fillna(0)
    industry_each_count=fund_df.groupby('申万分类').size()
    for idx in range(0, len(fund_df)):
        print(fund_df.iloc[idx]['品种代码'], fund_df.iloc[idx]['申万分类'])

        if fund_df.iloc[idx]['申万分类'] in industry_list:
            filefullpath = './tmp/' + fund_df.iloc[idx]['品种代码'] + fund_df.iloc[idx]['品种名称'] + '.csv'
            if os.path.isfile(filefullpath):
                stock_daily_df = pd.read_csv(filefullpath)
                stock_daily_df = stock_daily_df.tail(DATA_LENGTH)
                stock_daily_df["date"] = pd.to_datetime(stock_daily_df["date"]).dt.date
                stock_daily_df.set_index("date", inplace=True)
            else:
                if str(fund_df.iloc[idx]['品种代码'])[0] == '0' or str(fund_df.iloc[idx]['品种代码'])[0] == '3':
                    stock_daily_df = ak.stock_zh_index_daily(symbol='sz' + fund_df.iloc[idx]['品种代码'])
                else:
                    stock_daily_df = ak.stock_zh_index_daily(symbol='sh' + fund_df.iloc[idx]['品种代码'])
                #   stock_daily_df.index.name = 'date'
                stock_daily_df['date'] = stock_daily_df.index
                stock_daily_df = stock_daily_df.reset_index(drop=True)
                stock_daily_df["date"] = pd.to_datetime(stock_daily_df["date"]).dt.date
                stock_daily_df.set_index("date", inplace=True)
                ##Moving Average
                close = [float(x) for x in stock_daily_df['close']]
                stock_daily_df['MA20'] = talib.MA(np.array(close), timeperiod=20)
                stock_daily_df['MA60'] = talib.MA(np.array(close), timeperiod=60)
                stock_daily_df['MA200'] = talib.MA(np.array(close), timeperiod=200)
                stock_daily_df['EMA20'] = talib.EMA(np.array(close), timeperiod=20)
                stock_daily_df['EMA60'] = talib.EMA(np.array(close), timeperiod=60)
                stock_daily_df['EMA200'] = talib.EMA(np.array(close), timeperiod=200)
                stock_daily_df = stock_daily_df.fillna(0)

                stock_daily_df.to_csv(filefullpath)
                stock_daily_df = stock_daily_df.tail(DATA_LENGTH)
                # stock_daily_df["date"] = pd.to_datetime(stock_daily_df["date"]).dt.date

            stock_daily_df['num'] = stock_daily_df['close'] - stock_daily_df['MA20']
            stock_daily_df['num'] = stock_daily_df.num.apply(lambda x: 1 if x >= 0 else 0)
            count = len(stock_daily_df.index)
            padding_df = pd.DataFrame()
            while count < DATA_LENGTH:
                padding_df = padding_df.append(stock_daily_df.iloc[0])
                count = count + 1

            if len(stock_daily_df.index) < DATA_LENGTH:
                stock_daily_df = padding_df.append(stock_daily_df)

            industry_df[fund_df.iloc[idx]['申万分类']] = industry_df[fund_df.iloc[idx]['申万分类']].values + stock_daily_df['num'].values
    industry_df.rename(columns=lambda x: x + '(共' + str(industry_each_count[x]) + '支)', inplace=True)

    if name == '000300':
        industry_df.insert(loc=0, column='总股票（300PE百分位:' + pe_pb300[1] + "）",
                                     value=industry_df.sum(axis=1, skipna=True))
    else:
        industry_df.insert(loc=0, column='总股票（500PE百分位:' + pe_pb500[1] + "）",
                                     value=industry_df.sum(axis=1, skipna=True))
    # industry_df['沪深300'] = industry_df.sum(axis = 1, skipna = True)
    industry_df.index = stock_daily_df.index

    ##转置，横坐标是时间
    industry_df = industry_df.stack().unstack(0)

    # scaled_industry_20Fundamentals = (industry_df - industry_df.min(axis=0)) / (industry_df.max(axis=0)-industry_df.min(axis=0))
    scaled_industry_df = industry_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=99, l=55, sep=3, as_cmap=True)
    if name == '000300':
        industry_df = industry_df[industry_df.columns].astype(int)
        ax2 = sns.heatmap(scaled_industry_df, annot=industry_df, annot_kws={'size': 6}, cmap=rdgn, fmt='d',
                      linewidths=2, linecolor='black', xticklabels=True, cbar=False, ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=75, fontsize=6)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=10)
        ax2.yaxis.set_ticks_position('both')
        ax2.yaxis.tick_right()
        ax2.set_title("300行业宽度")
        # ax1.xaxis.set_label_position('top')

    if name == "000905":
        industry_df = industry_df[industry_df.columns].astype(int)
        ax3 = sns.heatmap(scaled_industry_df, annot=industry_df, annot_kws={'size': 6}, cmap=rdgn, fmt='d',
                      linewidths=2, linecolor='black', xticklabels=True, cbar=False, ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=75, fontsize=6)
        ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=10)
        ax3.yaxis.set_ticks_position('both')
        ax3.yaxis.tick_right()
        ax3.set_title("500行业宽度")
        # ax1.xaxis.set_label_position('top')

fig.savefig('行业宽度' + industry_daily_df.index[-1].strftime('%Y-%m-%d') + '.png', dpi=300)

#stock_margin_sse_df = ak.stock_margin_sse(start_date=industry_daily_df.index[0].strftime('%Y%m%d'), end_date=industry_daily_df.index[-1].strftime('%Y%m%d'))
#stock_margin_sse_df = stock_margin_sse_df.iloc[::-1]

print("end")
