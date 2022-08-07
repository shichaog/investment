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

import talib as ta
from finta import TA

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#may refer https://towardsdatascience.com/predicting-future-stock-market-trends-with-python-machine-learning-2bf3f1633b3c

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


"""
Defining some constants for data mining
"""
NUM_DAYS = 10000 #The number of trading days

#List of symbols for technical indicators
INDICATORS = ['RSI', 'MACD', 'STOCH', 'ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

"""
Get trading data by akshare python lib
"""

#list of stocks/ETFs used for training
Stock_list = {'300ETF':'sh510300'}
#Stock_list = {'300ETF':'sh510300','500ETF':'sh510500'}

#Get history trading data of pandas format by downloading or saved CSV
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
    daily_df = daily_df.rename(columns={'open': 'Open', 'high': 'High', 'close': 'Close', 'low': 'Low', 'volume': 'Volume'}, inplace=True)

    op = 'Open'
    cl = 'Close'
    hi = 'High'
    lo= 'Low'
    vo = 'Volume'

    #https://github.com/matplotlib/mplfinance/tree/master/tests
    daily_df.index = pd.DatetimeIndex(daily_df.index)
    daily_df.index.name = 'Date'

"""
smooth data so that model can learn from/;
"""
def _exponential_smooth(df, alpha):
    """
    make close price less 'rigid'
    :param df: pandas dataframe
    :param alpha: weight factor to weight recent values more
    :return:
    """
    return df.ewm(alpha=alpha).mean()

daily_df = _exponential_smooth(daily_df, 0.65)

def _get_indicator_data(df):
    """
    Functions that use the talib API to calculate technical indicators used as the fetures
    :param df:
    :return:
    """
    for indicator in INDICATORS:
        ind_data = eval('ta.' + indicator + '(df)')
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        df = df.merge(ind_data, left_index=True, right_index=True)
    df.rename(columns={"14 period EMV." : '14 period EMV'}, inplace=True)

    #calculate Moving averages as features
    df['ema50'] = df['Close']/df['Close'].ewm(50).mean()
    df['ema21'] = df['Close'] / df['Close'].ewm(21).mean()
    df['ema14'] = df['Close'] / df['Close'].ewm(14).mean()
    df['ema5'] = df['Close'] / df['Close'].ewm(5).mean()

    #instead of using the actual volume value, normalize it with a moving volume average
    df['normVol'] = df['volume'] / df['volume'].ewm(5).mean()

    # Remove columns that won't be used as features
    del (df['Open'])
    del (df['High'])
    del (df['Low'])
    del (df['Volume'])
    return df

daily_df = _get_indicator_data(daily_df, 0.65)

def _produce_prediction(df, window):
    """
    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
    :param df:
    :param window: number of days, or rows to look ahead to see what the price did
    :return:
    """
    prediction = (df.shift(-window)['Close'] >= df['Close'])
    prediction = prediction.iloc[:-window]
    df['pred'] = prediction.astype(int)

df = _produce_prediction(daily_df, window=15)
df = df.dropna() # Some indicators produce NaN values for the first few rows,

"""
Split data into traning and testing datasets
"""
def _train_random_forest(X_train, y_train, X_test, y_test):
    """
    use random forcast classifier to train the model
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    #create a new random forcast classifier
    rand_cfr = RandomForestClassifier()

    #Directory of all valuse we want to test for n_estimators
    params_rand_cfr = {'n_estimators': [110, 130, 140, 150, 160, 180, 200]}

    #use gridsearch to test all valuse for n_estimators
    rand_cfr_gs = GridSearchCV(rand_cfr, params_rand_cfr, cv=5)

    #Fit model to traning data
    rand_cfr_gs.fit(X_train, y_train)

    #Save best model
    rand_cfr_best = rand_cfr_gs.best_estimator_

    #check best n_estimators value
    print(rand_cfr_gs.best_params_)

    prediction = rand_cfr_best.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

    return rand_cfr_best

def _train_KNN(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 25)}

    # Use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)

    # Fit model to training data
    knn_gs.fit(X_train, y_train)

    # Save best model
    knn_best = knn_gs.best_estimator_

    # Check best n_neigbors value
    print(knn_gs.best_params_)

    prediction = knn_best.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

    return knn_best

def _ensemble_model(rf_model, knn_model, X_train, y_train, X_test, y_test):
    # Create a dictionary of our models
    estimators = [('knn', knn_model), ('rf', rf_model)]

    # Create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='hard')

    # fit model to training data
    ensemble.fit(X_train, y_train)

    # test our model on the test data
    print(ensemble.score(X_test, y_test))

    prediction = ensemble.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

    return ensemble


def cross_Validation(daily_df):
    # Split data into equal partitions of size len_train

    num_train = 10  # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    len_train = 40  # Length of each train-test set

    # Lists to store the results from each model
    rf_RESULTS = []
    knn_RESULTS = []
    ensemble_RESULTS = []

    i = 0
    while True:

        # Partition the df into chunks of size len_train every num_train days
        df = daily_df.iloc[i * num_train: (i * num_train) + len_train]
        i += 1
        print(i * num_train, (i * num_train) + len_train)

        if len(df) < 40:
            break

        y = df['pred']
        features = [x for x in df.columns if x not in ['pred']]
        X = df[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=7 * len(X) // 10, shuffle=False)

        rf_model = _train_random_forest(X_train, y_train, X_test, y_test)
        knn_model = _train_KNN(X_train, y_train, X_test, y_test)
        ensemble_model = _ensemble_model(rf_model, knn_model, X_train, y_train, X_test, y_test)

        rf_prediction = rf_model.predict(X_test)
        knn_prediction = knn_model.predict(X_test)
        ensemble_prediction = ensemble_model.predict(X_test)

        print('rf prediction is ', rf_prediction)
        print('knn prediction is ', knn_prediction)
        print('ensemble prediction is ', ensemble_prediction)
        print('truth values are ', y_test.values)

        rf_accuracy = accuracy_score(y_test.values, rf_prediction)
        knn_accuracy = accuracy_score(y_test.values, knn_prediction)
        ensemble_accuracy = accuracy_score(y_test.values, ensemble_prediction)

        print(rf_accuracy, knn_accuracy, ensemble_accuracy)
        rf_RESULTS.append(rf_accuracy)
        knn_RESULTS.append(knn_accuracy)
        ensemble_RESULTS.append(ensemble_accuracy)

    print('RF Accuracy = ' + str(sum(rf_RESULTS) / len(rf_RESULTS)))
    print('KNN Accuracy = ' + str(sum(knn_RESULTS) / len(knn_RESULTS)))
    print('Ensemble Accuracy = ' + str(sum(ensemble_RESULTS) / len(ensemble_RESULTS)))


cross_Validation(daily_df)


