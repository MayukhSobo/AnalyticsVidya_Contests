from pprint import pprint
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def xgb_model_fit(clf, features, labels, cv=True, folds=5, early_stopping=50):
	"""
	Fit the training dataset with XGBoost classifier
	mentioned by the clf. If cv=True, we perform cross-validation
	with param tuning
	"""

	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
	if cv:
		# For cross-validation
		xgb_param = clf.get_xgb_params()
		print("Running XGBoost with the following params:\n")
		pprint(xgb_param)
		xtrain = xgb.DMatrix(X_train, label=y_train)
		xgb_param_raw = clf.get_params()
		cvTrain = xgb.cv(xgb_param, xtrain,
			num_boost_round=xgb_param_raw['n_estimators'], 
			nfold=folds,
			metrics='error',  # Area under Curve
			early_stopping_rounds=early_stopping,
			verbose_eval=True,
			seed=42)
		clf.set_params(n_estimators=cvTrain.shape[0])

	# Fit the algorithm on the data
	clf.fit(X_train, y_train, eval_metric='error')

	# Predict training set
	pred = clf.predict(X_test)
	pred_prob = clf.predict_proba(X_test)[:, 1]

	# Print model report
	print("\nModel Report")
	print("Accuracy : {:.4f}".format(accuracy_score(y_test, pred)))
	print("AUC Score (CV): {:.4f}".format(roc_auc_score(y_test, pred_prob)))
	feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)
	df_imp = pd.DataFrame(feat_imp).reset_index()
	df_imp.columns = ['Feature', 'Score']
	return df_imp
