from xgboost.sklearn import XGBClassifier
# from xgb_model import xgb_model_fit
# from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
print('\n\n')

dtrain = pd.read_csv('../data/cleaned_train_v2.csv').iloc[:, 1:]
dtest = pd.read_csv('../data/cleaned_test_v2.csv').iloc[:, 1:]

# print(dtrain)
loan_ids = dtest.Loan_ID
dtest = dtest.iloc[:, 1:]

features = np.array(dtrain.iloc[:, 0:-1])
labels = np.array(dtrain.Loan_Status)
test = np.array(dtest)

# print(features.shape)
# Classifier 1 - XGBoost
clf1 = XGBClassifier(learning_rate=0.1, 
	n_estimators=1000,
	max_depth=3,
	min_child_weight=1,
	gamma=0.2,
	subsample=0.8,
	colsample_bytree=0.8,
	objective='binary:logistic',
	nthread=4,
	scale_pos_weight=1,
	reg_alpha=69,
	seed=42)

# Classifier 2 - Random Forest

clf2 = RandomForestClassifier(bootstrap=True,
	criterion='gini',
	max_depth=3,
	oob_score=True,
	max_features=3,
	min_samples_leaf=10,
	min_samples_split=10,
	random_state=42,
	n_jobs=-1)

# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.40, random_state=42)

tree = DecisionTreeClassifier(criterion='gini', max_depth=3)
clf3 = AdaBoostClassifier(base_estimator=tree, n_estimators=3000, learning_rate=0.03, random_state=42)

eclf = VotingClassifier(estimators=[
	('forest', clf2), 
	('xgboost', clf1),
	('adaboost', clf3)], 
	voting='hard')

eclf.fit(features, labels)
pred = eclf.predict(test)
# print(accuracy_score(y_test, pred))
# print("Random Forest Classifier.....")
# clf2.fit(X_train, y_train)
# pred1 = clf2.predict(X_test)
# print(accuracy_score(y_test, pred1))

# print('\nXGBoost Classifier......')
# clf1.fit(X_train, y_train)
# pred2 = clf1.predict(X_test)
# print(accuracy_score(y_test, pred2))
# clf1.fit(features, labels, eval_metric='error')
# pred = clf1.predict(test)
# print(pred)

submission = pd.DataFrame({'Loan_ID': loan_ids, 'Loan_Status': pred})
submission['Loan_Status'] = submission.Loan_Status.map({0: 'N', 1: 'Y'})
submission.to_csv('submission2.csv', index=False)
# xgb_model_fit(clf1, features, labels, folds=3)


# Classifier 2 - Random Forest


# Tuning ```max_depth``` and ```min_child_weight```

# param_dist = {
# 	"max_depth": list(range(1, 8, 2)),
# 	"max_features": list(range(1, 10, 2)),
# 	"min_samples_split": list(range(2, 11, 2)),
# 	"min_samples_leaf": list(range(2, 11, 2)),
# 	"bootstrap": [True, False],
# 	"criterion": ["gini", "entropy"]
# }


# gs1 = GridSearchCV(estimator=clf2,
# 	param_grid=param_dist,
# 	scoring='accuracy',
# 	n_jobs=-1,
# 	iid=False,
# 	cv=7,
# 	verbose=5)
# gs1.fit(features, labels)
# # bootstrap=True, criterion=gini, max_depth=3, max_features=3, min_samples_leaf=10, min_samples_split=10, score=0.870588
# # # print(gs1.grid_scores_)
# print(gs1.best_params_)
# print(gs1.best_score_)
