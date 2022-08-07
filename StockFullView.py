#!/usr/bin/env python
## -*- coding: utf-8 -*-
import collections
import json
from datetime import datetime, timedelta

import akshare as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

Stock_medical_dict = {
    # 创新药龙头
    '恒瑞医药': 'sh600276',
    # 全球产业链龙头
    '复星医药': 'sh600196',
    # 体外诊断试剂龙头
    '安图生物': 'sh603658',
    # 疫苗龙头
    '长春高新': 'sz000661',
    # 独门医疗服务
    '爱尔眼科': 'sz300015',
    '通策医疗': 'sh600763'}

Stock_ch_medical_dict = {
    # 中药
    '片仔癀': 'sh600436', '云南白药': 'sz000538',
    '以岭药业': 'sz002603', '同仁堂': 'sh600085',
    '葵花药业': 'sz002737', '东阿阿胶': 'sz000423',
    }
Stock_consum_dict = {
    '海天味业': 'sh603228', '中炬高新': 'sh600872', '百润股份': 'sz002568',
    '伊利股份': 'sh600887', '光明乳业': 'sh600597',
    '贵州茅台': 'sh600519', '五粮液': 'sz000858',
    }

Stock_indust_dict = {
    '美的集团': 'sz000333', '格力电器': 'sz000651',
    '比亚迪': 'SZ002594', '宁德时代':'SZ300750',
    '招商银行': 'sh600036',
}

def get_stock_financial_fundamental(code, name):

    stock_df = pd.DataFrame()
    # 每股收益率
    stock_analysis_df = ak.stock_financial_analysis_indicator(stock=code[2:])
    stock_analysis_df[stock_analysis_df == '--'] = np.NaN
    stock_analysis_df = stock_analysis_df[stock_analysis_df.index.str.contains('12-31')].head(4)
    stock_analysis_df = stock_analysis_df[['加权净资产收益率(%)', '资产负债率(%)', '扣除非经常性损益后的净利润(元)']]
    stock_analysis_df['扣除非经常性损益后的净利润(元)'] = stock_analysis_df["扣除非经常性损益后的净利润(元)"].astype(float)

    stock_df['资产负债率'] = round(stock_analysis_df['资产负债率(%)'].astype(float), 1)
    stock_df['净利润/亿元'] = round(stock_analysis_df['扣除非经常性损益后的净利润(元)'] / 100000000, 1)
    stock_df['ROE'] = round(stock_analysis_df['加权净资产收益率(%)'].astype(float), 1)

    #   stock_df['ALR'] = stock_analysis_df['资产负债率(%)'].astype(str)
    #   stock_df['NetProfit/亿元'] = stock_analysis_df['扣除非经常性损益后的净利润(元)'].astype(str)
    #   stock_df['ROE'] = stock_analysis_df['加权净资产收益率(%)'].astype(str)

    new_column_name = []
    new_values = []
    for financial_index in ['资产负债率', '净利润/亿元', 'ROE']:
        new_column_name.extend([year_index[0:4] + financial_index for year_index in stock_df.index])
        new_values.extend(stock_df[financial_index].to_list())

    stock_financial_dict = dict(zip(new_column_name, new_values))

    df = pd.DataFrame(stock_financial_dict, index=[name])

    stock_a_indicator_df = ak.stock_a_lg_indicator(stock=code[2:])
    sort_pe_df = stock_a_indicator_df.head(1200).sort_values(by="pe_ttm", ascending=True).reset_index(drop=True)
    pe = stock_a_indicator_df.loc[0, 'pe_ttm']
    index = sort_pe_df[sort_pe_df.pe_ttm == pe].index.tolist()[0]
    percent_pe = index/1200

    df['总市值亿元'] = stock_a_indicator_df.loc[0,'total_mv']
    df['总市值亿元'] = round(df['总市值亿元'].astype(float)/ 10000, 1)
    df['PE'] = stock_a_indicator_df.loc[0, 'pe_ttm']
    df['PE'] = round(df['PE'].astype(float), 2)
    df['5年PE百分位'] = round(percent_pe,2)

    return df,percent_pe,stock_a_indicator_df['trade_date'][0].strftime("%Y-%m-%d")

financial_fundamental_df = pd.DataFrame()
sorted_financial_fundamental_df = pd.DataFrame()
stock_pe_list = []
stock_name_list = []
last_date_index = ""

for stock_dict in [Stock_consum_dict,Stock_medical_dict, Stock_ch_medical_dict]:
    # for idx,row in hs300_df.iterrows():
    for name, symbol in stock_dict.items():
        financial_stock_df, PE_percent_pos, date_index = get_stock_financial_fundamental(symbol, name)
        if last_date_index.strip() == "":
            last_date_index = date_index
        else:
            if last_date_index != date_index:
                print("last date:%s vs date:%s"%(last_date_index, date_index) )
                exit(0)
        if financial_fundamental_df.empty:
            financial_fundamental_df = financial_stock_df
        else:
            financial_fundamental_df = pd.concat([financial_fundamental_df, financial_stock_df])

        stock_pe_list.append(PE_percent_pos)
        stock_name_list.append(name)
    financial_fundamental_df = financial_fundamental_df.sort_values(by="5年PE百分位")
    if sorted_financial_fundamental_df.empty:
        sorted_financial_fundamental_df = financial_fundamental_df
    else:
        sorted_financial_fundamental_df = pd.concat([sorted_financial_fundamental_df, financial_fundamental_df])

    financial_fundamental_df = pd.DataFrame()


df_norm_col = (sorted_financial_fundamental_df -sorted_financial_fundamental_df.mean()) /sorted_financial_fundamental_df.std()

fig = plt.figure(figsize=(16, 9), dpi=300)
heatmap = sns.heatmap(df_norm_col, annot=sorted_financial_fundamental_df, cmap='RdYlGn', fmt='g',
                      square=False, cbar=False, xticklabels=True, yticklabels=True,
                      linewidths=1)



wanted_index = sorted_financial_fundamental_df.index.to_list().index('恒瑞医药')

heatmap.add_patch(Rectangle((0, wanted_index), sorted_financial_fundamental_df.shape[1], 1, fill=False,
                              edgecolor='blue', lw=3, clip_on=False))

wanted_index = sorted_financial_fundamental_df.index.to_list().index('云南白药')
heatmap.add_patch(Rectangle((0, wanted_index), sorted_financial_fundamental_df.shape[1], 1, fill=False,
                              edgecolor='blue', lw=3, clip_on=False))

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=75)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
heatmap.tick_params(right=True, top=True, labelright=True, labeltop=True,)
plt.ylim(0,len(sorted_financial_fundamental_df))

#Add subtitle
plt.title('公司财报数据'+ date_index +' @shichaog', fontsize=12)

fig = plt.gcf()
plt.savefig('股票财务数据'+ date_index + '.png',
            bbox_inches='tight',
            edgecolor=fig.get_edgecolor(),
            facecolor=fig.get_facecolor(),
            dpi=300)




