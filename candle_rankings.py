# Note - 1
# Only some patterns have bull and bear versions. 
# However, to make the process unified and for codability purposes 
# all patterns are labeled with "_Bull" and "_Bear" tags.
# Both versions of the single patterns are given same performance rank, 
# since they will always return only 1 version.  

# Note - 2 
# Following TA-Lib patterns are excluded from the analysis since the corresponding ranking not found:
# CounterAttack, Longline, Shortline, Stalledpattern, Kickingbylength
# http://thepatternsite.com/

import talib
import numpy as np

candle_rankings = {
        "CDL3LINESTRIKE_Bull": 1,
        "CDL3LINESTRIKE_Bear": 2,
        "CDL3BLACKCROWS_Bull": 3,
        "CDL3BLACKCROWS_Bear": 3,
        "CDLEVENINGSTAR_Bull": 4,
        "CDLEVENINGSTAR_Bear": 4,
        "CDLTASUKIGAP_Bull": 5,
        "CDLTASUKIGAP_Bear": 5,
        "CDLINVERTEDHAMMER_Bull": 6,
        "CDLINVERTEDHAMMER_Bear": 6,
        "CDLMATCHINGLOW_Bull": 7,
        "CDLMATCHINGLOW_Bear": 7,
        "CDLABANDONEDBABY_Bull": 8,
        "CDLABANDONEDBABY_Bear": 8,
        "CDLBREAKAWAY_Bull": 10,
        "CDLBREAKAWAY_Bear": 10,
        "CDLMORNINGSTAR_Bull": 12,
        "CDLMORNINGSTAR_Bear": 12,
        "CDLPIERCING_Bull": 13,
        "CDLPIERCING_Bear": 13,
        "CDLSTICKSANDWICH_Bull": 14,
        "CDLSTICKSANDWICH_Bear": 14,
        "CDLTHRUSTING_Bull": 15,
        "CDLTHRUSTING_Bear": 15,
        "CDLINNECK_Bull": 17,
        "CDLINNECK_Bear": 17,
        "CDL3INSIDE_Bull": 20,
        "CDL3INSIDE_Bear": 56,
        "CDLHOMINGPIGEON_Bull": 21,
        "CDLHOMINGPIGEON_Bear": 21,
        "CDLDARKCLOUDCOVER_Bull": 22,
        "CDLDARKCLOUDCOVER_Bear": 22,
        "CDLIDENTICAL3CROWS_Bull": 24,
        "CDLIDENTICAL3CROWS_Bear": 24,
        "CDLMORNINGDOJISTAR_Bull": 25,
        "CDLMORNINGDOJISTAR_Bear": 25,
        "CDLXSIDEGAP3METHODS_Bull": 27,
        "CDLXSIDEGAP3METHODS_Bear": 26,
        "CDLTRISTAR_Bull": 28,
        "CDLTRISTAR_Bear": 76,
        "CDLGAPSIDESIDEWHITE_Bull": 46,
        "CDLGAPSIDESIDEWHITE_Bear": 29,
        "CDLEVENINGDOJISTAR_Bull": 30,
        "CDLEVENINGDOJISTAR_Bear": 30,
        "CDL3WHITESOLDIERS_Bull": 32,
        "CDL3WHITESOLDIERS_Bear": 32,
        "CDLONNECK_Bull": 33,
        "CDLONNECK_Bear": 33,
        "CDL3OUTSIDE_Bull": 34,
        "CDL3OUTSIDE_Bear": 39,
        "CDLRICKSHAWMAN_Bull": 35,
        "CDLRICKSHAWMAN_Bear": 35,
        "CDLSEPARATINGLINES_Bull": 36,
        "CDLSEPARATINGLINES_Bear": 40,
        "CDLLONGLEGGEDDOJI_Bull": 37,
        "CDLLONGLEGGEDDOJI_Bear": 37,
        "CDLHARAMI_Bull": 38,
        "CDLHARAMI_Bear": 72,
        "CDLLADDERBOTTOM_Bull": 41,
        "CDLLADDERBOTTOM_Bear": 41,
        "CDLCLOSINGMARUBOZU_Bull": 70,
        "CDLCLOSINGMARUBOZU_Bear": 43,
        "CDLTAKURI_Bull": 47,
        "CDLTAKURI_Bear": 47,
        "CDLDOJISTAR_Bull": 49,
        "CDLDOJISTAR_Bear": 51,
        "CDLHARAMICROSS_Bull": 50,
        "CDLHARAMICROSS_Bear": 80,
        "CDLADVANCEBLOCK_Bull": 54,
        "CDLADVANCEBLOCK_Bear": 54,
        "CDLSHOOTINGSTAR_Bull": 55,
        "CDLSHOOTINGSTAR_Bear": 55,
        "CDLMARUBOZU_Bull": 71,
        "CDLMARUBOZU_Bear": 57,
        "CDLUNIQUE3RIVER_Bull": 60,
        "CDLUNIQUE3RIVER_Bear": 60,
        "CDL2CROWS_Bull": 61,
        "CDL2CROWS_Bear": 61,
        "CDLBELTHOLD_Bull": 62,
        "CDLBELTHOLD_Bear": 63,
        "CDLHAMMER_Bull": 65,
        "CDLHAMMER_Bear": 65,
        "CDLHIGHWAVE_Bull": 67,
        "CDLHIGHWAVE_Bear": 67,
        "CDLSPINNINGTOP_Bull": 69,
        "CDLSPINNINGTOP_Bear": 73,
        "CDLUPSIDEGAP2CROWS_Bull": 74,
        "CDLUPSIDEGAP2CROWS_Bear": 74,
        "CDLGRAVESTONEDOJI_Bull": 77,
        "CDLGRAVESTONEDOJI_Bear": 77,
        "CDLHIKKAKEMOD_Bull": 82,
        "CDLHIKKAKEMOD_Bear": 81,
        "CDLHIKKAKE_Bull": 85,
        "CDLHIKKAKE_Bear": 83,
        "CDLENGULFING_Bull": 84,
        "CDLENGULFING_Bear": 91,
        "CDLMATHOLD_Bull": 86,
        "CDLMATHOLD_Bear": 86,
        "CDLHANGINGMAN_Bull": 87,
        "CDLHANGINGMAN_Bear": 87,
        "CDLRISEFALL3METHODS_Bull": 94,
        "CDLRISEFALL3METHODS_Bear": 89,
        "CDLKICKING_Bull": 96,
        "CDLKICKING_Bear": 102,
        "CDLDRAGONFLYDOJI_Bull": 98,
        "CDLDRAGONFLYDOJI_Bear": 98,
        "CDLCONCEALBABYSWALL_Bull": 101,
        "CDLCONCEALBABYSWALL_Bear": 101,
        "CDL3STARSINSOUTH_Bull": 103,
        "CDL3STARSINSOUTH_Bear": 103,
        "CDLDOJI_Bull": 104,
        "CDLDOJI_Bear": 104,
        "CDLSHORTLINE_Bull":105,
        "CDLSHORTLINE_Bear":105,
        "CDLLONGLINE_Bull":106,
        "CDLLONGLINE_Bear":106,
        "CDLSTALLEDPATTERN_Bear":107,
        "CDLSTALLEDPATTERN_Bull":107,
        "CDLCOUNTERATTACK_Bull": 108,
        "CDLCOUNTERATTACK_Bear": 108

    }

candle_names = [
    'CDLHAMMER',
    'CDLINVERTEDHAMMER',
    'CDLHANGINGMAN',
    'CDLSHOOTINGSTAR'
]

def find_candle_pattern(daily_df):
        op = daily_df["Open"]
        hi = daily_df["High"]
        lo = daily_df["Low"]
        cl = daily_df["Close"]
        ## find patterns
        #candle_names = talib.get_function_groups()['Pattern Recognition']
        # create columns for each pattern
        for candle in candle_names:
                # below is same as;
                # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
                daily_df[candle] = getattr(talib, candle)(op, hi, lo, cl)
        daily_df['candlestick_pattern'] = np.nan
        daily_df['candlestick_match_count'] = np.nan

        for index, row in daily_df.iterrows():
                # no pattern found
                if len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
                        daily_df.loc[index, 'candlestick_pattern'] = "NO_PATTERN"
                        daily_df.loc[index, 'candlestick_match_count'] = 0
                # single pattern found
                elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
                        row__ = row[candle_names]
                        row__key__ = row[candle_names].keys()
                        row__value__ = row[candle_names].values
                        # bull pattern 100 or 200
                        if any(row[candle_names].values > 0):
                                pattern = list(np.compress(row[candle_names].values != 0, row[candle_names].keys()))[
                                                  0]
                                daily_df.loc[index, 'candlestick_pattern'] = pattern
                                daily_df.loc[index, 'candlestick_match_count'] = 1
                        # bear pattern -100 or -200
                        else:
                                pattern = list(np.compress(row[candle_names].values != 0, row[candle_names].keys()))[
                                                  0]
                                daily_df.loc[index, 'candlestick_pattern'] = pattern
                                daily_df.loc[index, 'candlestick_match_count'] = 1
                # multiple patterns matched -- select best performance
                else:
                        # filter out pattern names from bool list of values
                        patterns = list(np.compress(row[candle_names].values != 0, row[candle_names].keys()))
                        container = []
                        for pattern in patterns:
                                if row[pattern] > 0:
                                        container.append(pattern + '_Bull')
                                else:
                                        container.append(pattern + '_Bear')
                        rank_list = [candle_rankings[p] for p in container]
                        if len(rank_list) == len(container):
                                rank_index_best = rank_list.index(min(rank_list))
                                daily_df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                                daily_df.loc[index, 'candlestick_match_count'] = len(container)
        daily_df.drop(candle_names, axis=1, inplace=True)

        return daily_df


