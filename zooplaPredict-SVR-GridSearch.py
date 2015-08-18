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
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score (MSE)")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, scoring="mean_squared_error", cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores*-1, axis=1)
    train_scores_std = np.std(train_scores*-1, axis=1)
    test_scores_mean = np.mean(test_scores*-1, axis=1)
    test_scores_std = np.std(test_scores*-1, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")

    print train_scores
    print test_scores
    print train_scores_mean
    print test_scores_mean

    plt.show(block=False)


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

cv = cross_validation.KFold(len(X), n_folds=3, shuffle=True, random_state=None)

# for i in range(1, 2):
# 	estimator = svm.SVR(kernel='poly', C=1)
# 	title = "learning curve (k = " + str(i) + ", cv = 3)"
# 	plot_learning_curve(estimator, title, X_shuf, Y_shuf, ylim=(0.0, 1.01), cv=cv, n_jobs=4)

# input("Press Enter to continue...")


# cross validation (just spliting the dataset)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_shuf, Y_shuf, test_size=0.2, random_state=0)

# printing header
#newHeader = header_not_num + vec.get_feature_names()
#print newHeader

#C_range = np.logspace(-2, 10, 7)
#gamma_range = np.logspace(-9, 3, 7)

# param_grid = {'kernel': ['poly', 'rbf'],
# 			  'C': [0.1, 1, 10, 100, 1000], #C_range, 
#  			  'degree': [2, 3, 4, 5, 6], 
#  			  'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001]}
# specify parameters and distributions to sample from
param_dist = {'kernel': ['poly'],
			  'C': [0.1, 1, 10, 100], #C_range, 
 			  'degree': [2, 3], 
 			  'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001]}


print "\n\n=== Reuslt of SVR ==="

result = {}	
result["name"] = "SVR"

# iter = 1 took 141.05 seconds
# iter = 30 -> 70 minutes?
n_iter_search = 5
clf = svm.SVR(cache_size=1000)
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, 
									n_iter=n_iter_search, scoring="mean_squared_error", n_jobs=-1, verbose=5)

from time import time
start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
#report(random_search.grid_scores_)
print random_search.grid_scores_
print random_search.best_estimator_
print random_search.best_score_
print random_search.best_params_




# import time
# start = time.time()
# gs_cv = GridSearchCV(clf, param_grid, scoring="mean_squared_error", n_jobs=-1).fit(X_train, y_train)
# end = time.time()
# print "Grid Search time (sec):"
# print end - start

# print gs_cv.grid_scores_
# print gs_cv.best_estimator_
# print gs_cv.best_score_
# print gs_cv.best_params_
# print gs_cv.scorer_

