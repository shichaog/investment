# invest
这个仓库用于资产管理和投资，投资品种诸如，银行理财、基金、股票、信托以及私募等权益类资产，目的是实现风险偏好和收益偏好的均衡、

## Environment
The code was developed using python 3 on MacOS.But the latest version would be okay.

## How to run
* 0.前置依赖
```
安装talib的依赖：https://mrjbq7.github.io/ta-lib/install.html
```

* 1.Install dependencies
```
pip3 install -r requirements.txt
```
需要手动创建tmp目录，在终端下mkdir tmp

* 2.Generate Industry fundamentals
```
说明：行业基本面 based on 沪深300 stocks.
输出：png图片，生成文件命名如《沪深300行业宽度2021-05-20.png》
注意事项：基本面用到融资融余额，在次日8点才会更新，缺少融资融券余额信息会导致生成png失败
python3 industry_fundamental.py
```
*  3.Generate stock analysis
```
说明： Techinal analysis stocks, 目前关注消费、医药以及相关ETF基金。
输出：Each .png represents a stock(each stock now list as a dict in TA_stock.py file ), include trand line, candle stick pattern recognition.
用到的子文件：calBuyPoint.py 买卖点计算.；candle_rankings.py 基于蜡烛图（K线）的牛熊识别，是比趋势线更早反馈趋势变换的技术指标，可配合趋势线使用；
python3 TA_stock.py
```
* 4. Add stock financial support
python3 StockFullView.py

## License
All rights reversed. shichaog@126.com 请勿转载，谢谢




