"""
After boosting for 1000 rounds we have got 
an improvement on RMSE as following

train-rmse:1860.04	test-rmse:1948.63

from 

train-rmse:2038.93	test-rmse:2099.68
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

def xgb_cross_validation(features, labels, num_rounds, cols='*', split_ratio=0.4):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        labels, 
                                                        test_size=split_ratio, 
                                                        random_state=42)
    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.01
    param['max_depth'] = 9
    param['seed'] = 42
    param['nthread'] = -1
    param['eval_metric'] = "rmse"
    param['silent'] = 1
    param['min_child_weight'] = 4
    param['colsample_bytree'] = 0.8
    param['colsample_bylevel'] = 0.8
    param['n_estimators'] = 200
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(param, dtrain, num_rounds, watchlist)
#     pred = model.predict(dtest)
    return

data = pd.read_csv('data/feature_set_3.csv').iloc[:, 1:]

features = data.iloc[:, 0:-1]
labels = data.Purchase
xgb_cross_validation(features, labels, num_rounds=1000, split_ratio=0.3)
