import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, preprocessing, svm, tree, \
		linear_model, cross_validation, naive_bayes, neighbors, \
		gaussian_process, cross_decomposition, ensemble, metrics
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve

## Constant Variables ##
csvfile = 'rentList_E_EC_N_NW_SE_WC_final.csv'

## Functions ##
def readCsvIntoPandasDataframe(csvfile):
	return pd.read_csv(csvfile)

def readCsvIntoDict(csvfile):
	data = []
	with open(csvfile) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data.append(row)
	return data


def displayResults(model, X_train, X_test, y_train, y_test):
	print "\n\n=== Reuslt of ", model["name"], " ==="

	result = {}	
	result["name"] = model["name"]

	clf = model["model"]
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred_sample = scaler.inverse_transform(y_pred[:20])
	y_test_sample = scaler.inverse_transform(y_test[:20])
	
	print "\npredicted result, true result"
	for i in range(len(y_test_sample)):
		print y_pred_sample[i], "\t", y_test_sample[i]

	result = calculateScores(y_pred, y_test, result)

	print clf.feature_importances_


def predictAndExport(models, X_train, X_test, y_train, y_test):
	results = []
	fieldnames = ["name", "mean_squared_error", "mean_absolute_error", "median_absolute_error", "r2", "explained_variance_score"]
	fout = open("result.csv", 'w')
	writer = csv.DictWriter(fout, fieldnames = fieldnames)
	writer.writeheader()

	for model in models:
		print "\n\n=== Reuslt of ", model["name"], " ==="

		result = {}	
		result["name"] = model["name"]

		clf = model["model"]
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		y_pred_sample = scaler.inverse_transform(y_pred[:20])
		y_test_sample = scaler.inverse_transform(y_test[:20])
	
		print "\npredicted result, true result"
		for i in range(len(y_test_sample)):
			print y_pred_sample[i], "\t", y_test_sample[i]

		result = calculateScores(y_pred, y_test, result)
		writer.writerow(result)

	return results

def calculateScores(y_pred, y_test, result):
	result["mean_squared_error"] = metrics.mean_squared_error(y_pred, y_test) 
	print "mean_squared_error:", result["mean_squared_error"]
		
	result["mean_absolute_error"] = metrics.mean_absolute_error(y_pred, y_test) 
	print "mean_absolute_error:", result["mean_absolute_error"]
	
	result["median_absolute_error"] = metrics.median_absolute_error(y_pred, y_test) 
	print "median_absolute_error:", result["median_absolute_error"]

	result["r2"] = metrics.r2_score(y_pred, y_test)
	print "r2_score:", result["r2"]

	result["explained_variance_score"] = metrics.explained_variance_score(y_pred, y_test)
	print "explained_variance_score:", result["explained_variance_score"]

	return result


# Import CSV
data = readCsvIntoPandasDataframe(csvfile)

# extract categorical columns (cat = categorical)
header_not_cat = list(data.columns.values)
header_not_cat.remove('property_type')
X_cat = data.drop( header_not_cat, axis = 1 )

# convert to dict
X_cat.fillna( 'NA', inplace = True )
X_cat = X_cat.T.to_dict().values() 

# vectorize categorical feature
vec = DV( sparse = False )
X_cat = vec.fit_transform(X_cat) 
print vec.get_feature_names()

# extract numerical columns (num = numerical)
header_not_num = list(data.columns.values)
header_not_num.remove('property_type')
header_not_num.remove('price')
X_num = data.drop( ['property_type','price'], axis = 1 )
X_num = X_num.values # pandas dataframe to numpy array

# impute n/a value (replace it with mean value)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_num = imp.fit_transform(X_num)

# scale the data
X_num = preprocessing.scale(X_num)

# combine numerical data and vectorized categorical data
X = np.hstack((X_num, X_cat))

# extract label column (predicted values: price)
header_features = list(data.columns.values)
header_features.remove('price')
Y = data.drop( header_features, axis = 1)
Y = Y.price

# standardize label column
scaler = preprocessing.StandardScaler().fit(Y)
Y = scaler.transform(Y)

# shuffle the dataset
from random import shuffle
index_shuf = range(len(X))
shuffle(index_shuf)
X_shuf = [X[i] for i in index_shuf]
Y_shuf = [Y[i] for i in index_shuf]

# cross validation (just spliting the dataset)
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_shuf, Y_shuf, test_size=0.2, random_state=0)

# printing header
newHeader = header_not_num + vec.get_feature_names()
#print newHeader


print "\n\n=== Reuslt of Random Forest ==="

result = {}	
result["name"] = "Random Forest"


cv = cross_validation.KFold(len(X), n_folds=10, shuffle=True, random_state=None)
ylim=(0.0, 1.01)
title = "Random Forest Validation Curve (cv = 3, max_features = 0.1)"


#param_range = np.linspace(10, 20, 2, dtype = 'int').tolist() 

param_range = [1] + np.linspace(10, 90, 9, dtype = 'int').tolist() \
			      + np.linspace(100, 1500, 15, dtype = 'int').tolist()
print param_range


clf = ensemble.RandomForestRegressor(max_features = 0.05)

import time
start = time.time()
train_scores, test_scores = validation_curve(
    clf, X_shuf, Y_shuf, param_name="n_estimators", param_range=param_range,
    cv=3, scoring="mean_squared_error", n_jobs=-1, verbose=5)
print("RandomizedSearchCV took %.2f seconds." % (time.time() - start) )

plt.figure()
plt.title(title)
if ylim is not None:
    plt.ylim(*ylim)
plt.xlabel("# of trees")
plt.ylabel("Score (MSE)")

train_scores_mean = np.mean(train_scores*-1, axis=1)
train_scores_std = np.std(train_scores*-1, axis=1)
test_scores_mean = np.mean(test_scores*-1, axis=1)
test_scores_std = np.std(test_scores*-1, axis=1)
plt.grid()

print "train_scores:"
print train_scores_mean
print "test_scores:"
print test_scores_mean

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")

plt.show()

# == show feature importances later !! ==
#print clf.feature_importances_


## 1.11. Ensemble methods
# averaging methods
#models.append( {"name": "1.11.4. Gradient Tree Boosting", \
#				"model": ensemble.GradientBoostingRegressor()} )
#displayResults(models[0], X_train, X_test, y_train, y_test)
#results = predictAndExport(models, X_train, X_test, y_train, y_test)

