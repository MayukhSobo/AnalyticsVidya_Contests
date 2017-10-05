import pandas as pd
import numpy as np
from math import sqrt # Because math.sqrt seems fastest
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

data = pd.read_csv('data/feature_set_3.csv').iloc[:, 1:] # Because this gave best performance

def RMSE(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return sqrt(mse)

root_mean_square_error = make_scorer(RMSE, greater_is_better=False)

param = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [2,3,4,5,6,7,8,9],
    'min_child_weight': [2,3,4,5],
    'colsample_bytree': [0.2,0.6,0.8],
    'colsample_bylevel': [0.2,0.6,0.8]
}
reg = XGBRegressor(
    objective="reg:linear",
    seed=42
)
grid = GridSearchCV(
    estimator = reg,
    param_grid = param,
    scoring=root_mean_square_error,
    cv = 7,
    verbose=100,
    n_jobs=4
)

features = data.iloc[:, 0:-1].as_matrix()
labels = data.Purchase.as_matrix()

grid.fit(features, labels)
print(grid.best_params_) # {'colsample_bylevel': 0.8, 'colsample_bytree': 0.8, 'max_depth': 9, 'min_child_weight': 4, 'n_estimators': 200}
print(grid.best_score_) # Sign flipped: -1799.0593057