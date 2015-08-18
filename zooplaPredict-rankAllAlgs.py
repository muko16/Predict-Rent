import json
import csv
import numpy as np
import pandas as pd
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


def displayResults(clf, title):
	print "\n\n=== Reuslt of ", title, " ==="
	y_pred_raw = clf.predict(X_test)
	y_pred = scaler.inverse_transform(y_pred_raw[:20])
	y_true_raw = y_test
	y_true = scaler.inverse_transform(y_true_raw[:20])

	print "\npredicted result, true result"
	for i in range(len(y_true)):
		print y_pred[i], "\t", y_true[i]

	print "\nr2_score:"
	print r2_score(y_true_raw, y_pred_raw)  
	print "\nexplained_variance_score:"
	print explained_variance_score(y_true_raw, y_pred_raw) 
	print "\nmean_squared_error:"
	print mean_squared_error(y_true_raw, y_pred_raw) 


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

		writer.writerow(result)

	return results

## main function starts here ##
#print "version = ", sl.__version__

# Import CSV
data = readCsvIntoPandasDataframe(csvfile)
#data = readCsvIntoDict(csvfile)


# extract categorical columns (cat = categorical)
header_not_cat = list(data.columns.values)
header_not_cat.remove('property_type')
X_cat = data.drop( header_not_cat, axis = 1 )
#print X_cat

# convert to dict
X_cat.fillna( 'NA', inplace = True )
X_cat = X_cat.T.to_dict().values() 
#print X_cat_val

# vectorize categorical feature
vec = DV( sparse = False )
X_cat = vec.fit_transform(X_cat) 
#print vec.get_feature_names()
#print vec_x_cat[:1]


# extract numerical columns (num = numerical)
header_not_num = list(data.columns.values)
header_not_num.remove('property_type')
header_not_num.remove('price')
X_num = data.drop( ['property_type','price'], axis = 1 )
#print X_num[:1]
X_num = X_num.values # pandas dataframe to numpy array

print X_num[9437]


imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_num = imp.fit_transform(X_num)
print X_num[9437]

# scale the data
#print X_num[0]
X_num = preprocessing.scale(X_num)
#print X_num[0]


# combine numerical data and vectorized categorical data
#print X_num[:1]
#print X_cat[:1]
X = np.hstack((X_num, X_cat))
#print X[:1]



# extract label column (predicted values: price)
header_features = list(data.columns.values)
header_features.remove('price')
Y = data.drop( header_features, axis = 1)
Y = Y.price

# standardize label column
scaler = preprocessing.StandardScaler().fit(Y)
#print Y[:1]
Y = scaler.transform(Y)
#print Y[:1]
#Y = scaler.inverse_transform(Y)
#print Y[:1]


# need to shuffle?
from random import shuffle
index_shuf = range(len(X))
shuffle(index_shuf)
X_shuf = [X[i] for i in index_shuf]
Y_shuf = [Y[i] for i in index_shuf]

# Dimensionaly reduction?

# model selection?


# cross validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_shuf, Y_shuf, test_size=0.2, random_state=0)

#print X_train.shape, y_train.shape
#print X_test.shape, y_test.shape


## SVR ##
#clf = svm.SVR(kernel='rbf', C=1).fit(X_train, y_train)
#clf = svm.SVR(kernel='rbf', C=1)
#displayResults(clf, "SVR rbf")

#kfold = cross_validation.KFold( len(X), n_folds=10, shuffle=True )
#result = [ clf.fit( X[train], Y[train] ).score( X[test], Y[test] ) for train, test in kfold ]
#print result

#scores = cross_validation.cross_val_score(clf, X, Y, cv=kfold, n_jobs=-1)
#print scores



models = []

# Linear Regression
## 1.1. Generalized Linear Models
models.append( {"name": "1.1.1. Ordinary Least Squares", \
				"model": linear_model.LinearRegression() } )
models.append( {"name": "1.1.2. Ridge Regression", \
				"model": linear_model.Ridge(alpha = .5)} )
models.append( {"name": "1.1.3. Lasso", \
				"model": linear_model.Lasso()} )
models.append( {"name": "1.1.4. Elastic Net", \
				"model": linear_model.ElasticNetCV()} )
## ValueError: For mono-task outputs, use ElasticNet
#models.append( {"name": "1.1.5. Multi-task Lasso", \
#				"model": linear_model.MultiTaskLasso()} )
## for high-dimensional data. maybe not suitable for this data.
#models.append( {"name": "1.1.6. Least Angle Regression", \
#				"model": linear_model.Lars()} )
models.append( {"name": "1.1.7. LARS Lasso", \
				"model": linear_model.LassoLars()} )
models.append( {"name": "1.1.8. Orthogonal Matching Pursuit (OMP)", \
				"model": linear_model.OrthogonalMatchingPursuit()} )
models.append( {"name": "1.1.9.1. Bayesian Ridge Regression", \
				"model": linear_model.BayesianRidge()} )

## Doesn't finish?
#models.append( {"name": "1.1.9.2. Automatic Relevance Determination - ARD", \
#				"model": linear_model.ARDRegression()} )

## Doesn't finish?
#models.append( {"name": "1.1.10. Logistic regression", \
#				"model": linear_model.LogisticRegression()} )

models.append( {"name": "1.1.11. Stochastic Gradient Descent - SGD", \
				"model": linear_model.SGDRegressor()} )

## ValueError("Unknown label type: %r" % ys)
#models.append( {"name": "1.1.12. Perceptron", \
#				"model": linear_model.Perceptron()} )
models.append( {"name": "1.1.13. Passive Aggressive Algorithms", \
				"model": linear_model.PassiveAggressiveRegressor()} )

models.append( {"name": "1.1.14.2. RANSAC: RANdom SAmple Consensus", \
				"model": linear_model.RANSACRegressor()} )
models.append( {"name": "1.1.14.3. Theil-Sen estimator: generalized-median-based estimator", \
				"model": linear_model.TheilSenRegressor()} )

## doesn't finish?
# fit to an order-3 polynomial data
#models.append( {"name": "1.1.15. Polynomial regression", \
#				"model": Pipeline([('poly', PolynomialFeatures(degree=3)), \
#							('linear', linear_model.LinearRegression(fit_intercept=False))]) } )


## 1.2. Linear and quadratic discriminant analysis
## Classifiers


## 1.3. Kernel ridge regression
# slow
#models.append( {"name": "1.3. Kernel ridge regression", \
#				"model": KernelRidge()} )


## 1.4. Support Vector Machines
## 1.4.2. Regression
models.append( {"name": "1.4.2 SVR rbf", \
				"model": svm.SVR(kernel='rbf', C=1)} )
models.append( {"name": "1.4.2 SVR linear", \
				"model": svm.SVR(kernel='linear', C=1)} )
models.append( {"name": "1.4.2 SVR sigmoid", \
				"model": svm.SVR(kernel='sigmoid', C=1)} )
models.append( {"name": "1.4.2 SVR poly", \
				"model": svm.SVR(kernel='poly', C=1)} )

## NuSVR ##
# slow
#models.append( {"name": "1.4.2 NuSVR rbf", \
#				"model": svm.NuSVR(kernel='rbf', C=1)} )
#models.append( {"name": "1.4.2 NuSVR linear", \
#				"model": svm.NuSVR(kernel='linear', C=1)} )
#models.append( {"name": "1.4.2 NuSVR sigmoid", \
#				"model": svm.NuSVR(kernel='sigmoid', C=1)} )
models.append( {"name": "1.4.2 NuSVR poly", \
				"model": svm.NuSVR(kernel='poly', C=1)} )

## LinearSVR ##
models.append( {"name": "1.4.2 LinearSVR", \
				"model": svm.LinearSVR(C=1, loss="epsilon_insensitive")} )


## 1.5. Stochastic Gradient Descent
## 1.5.2. Regression
## refer to 1.1.11. Stochastic Gradient Descent - SGD


## 1.6. Nearest Neighbors 
models.append( {"name": "1.6.3. KNeighborsRegressor uniform", \
				"model": neighbors.KNeighborsRegressor(weights = "uniform")} )
models.append( {"name": "1.6.3. KNeighborsRegressor distance", \
				"model": neighbors.KNeighborsRegressor(weights = "distance")} )


#ValueError: Input contains NaN
#models.append( {"name": "1.6.3. RadiusNeighborsRegressor uniform", \
#				"model": neighbors.RadiusNeighborsRegressor(weights = "uniform")} )
#ZeroDivisionError: Weights sum to zero, can't be normalized
#models.append( {"name": "1.6.3. RadiusNeighborsRegressor distance", \
#				"model": neighbors.RadiusNeighborsRegressor(weights = "distance")} )
models.append( {"name": "1.6.3. NearestCentroid", \
				"model": neighbors.NearestCentroid()} )


## 1.7. Gaussian Processes
## too slow?
#models.append( {"name": "1.7. Gaussian Processes", \
#				"model": gaussian_process.GaussianProcess()} )


## 1.8. Cross decomposition
models.append( {"name": "1.8. Cross decomposition PLSRegression", \
				"model": cross_decomposition.PLSRegression()} )
models.append( {"name": "1.8. Cross decomposition PLSCanonical", \
				"model": cross_decomposition.PLSCanonical()} )
# slow
#models.append( {"name": "1.8. Cross decomposition CCA", \
#				"model": cross_decomposition.CCA()} )


## 1.9. Naive Bayes (for classification?)
#ValueError: Unknown label type: array
#models.append( {"name": "1.9.1. GaussianNB", \
#				"model": naive_bayes.GaussianNB()} )

# doesn't work for this dataset?
#models.append( {"name": "1.9.2. MultinomialNB", \
#				"model": naive_bayes.MultinomialNB()} )

# doesn't work for this dataset?
#models.append( {"name": "1.9.3. BernoulliNB", \
#				"model": naive_bayes.BernoulliNB()} )


## 1.10. Decision Trees
models.append( {"name": "1.10. DecisionTreeRegressor", \
				"model": tree.DecisionTreeRegressor(random_state=0)} )
models.append( {"name": "1.10. ExtraTreeRegressor", \
				"model": tree.ExtraTreeRegressor(random_state=0)} )


## 1.11. Ensemble methods
# averaging methods
models.append( {"name": "1.11.1. Bagging meta-estimator", \
				"model": ensemble.BaggingRegressor(neighbors.KNeighborsRegressor())} )
models.append( {"name": "1.11.2.1. Random Forests", \
				"model": ensemble.RandomForestRegressor()} )
models.append( {"name": "1.11.2.2. Extremely Randomized Trees", \
				"model": ensemble.ExtraTreesRegressor()} )
models.append( {"name": "1.11.3. AdaBoost", \
				"model": ensemble.AdaBoostRegressor()} )
models.append( {"name": "1.11.4. Gradient Tree Boosting", \
				"model": ensemble.GradientBoostingRegressor()} )


## 1.12. Multiclass and multilabel algorithms
# not regression 

## 1.13. Feature selection
# not about estimator

## 1.14. Semi-Supervised
# all samples have price data, so doesn't apply


## 1.15. Isotonic regression
# ValueError("X should be a 1d array")
#models.append( {"name": "1.14. Semi-Supervised", \
#				"model": IsotonicRegression()} )


results = predictAndExport(models, X_train, X_test, y_train, y_test)









# Linear Regression
## 1.1. Generalized Linear Models
## 1.1.1. Ordinary Least Squares
# clf = linear_model.LinearRegression().fit(X_train, y_train)
# displayResults(clf, "LinearRegression")

## 1.1.2. Ridge Regression
# clf = linear_model.Ridge(alpha = .5).fit(X_train, y_train)
# displayResults(clf, "Ridge")

## 1.1.3. Lasso
# clf = linear_model.Lasso().fit(X_train, y_train)
# displayResults(clf, "Lasso")

## 1.1.4. Elastic Net
# clf = linear_model.ElasticNetCV().fit(X_train, y_train)
# displayResults(clf, "ElasticNetCV")

## 1.1.5. Multi-task Lasso
## ValueError: For mono-task outputs, use ElasticNet
## clf = linear_model.MultiTaskLasso().fit(X_train, y_train)
## displayResults(clf, "Multi-task Lasso")

## 1.1.6. Least Angle Regression
## for high-dimensional data. maybe not suitable for this data.
# clf = linear_model.Lars().fit(X_train, y_train)
# displayResults(clf, "Lars")

## 1.1.7. LARS Lasso
#clf = linear_model.LassoLars().fit(X_train, y_train)
#displayResults(clf, "Lars Lasso")

## 1.1.8. Orthogonal Matching Pursuit (OMP)
#clf = linear_model.OrthogonalMatchingPursuit().fit(X_train, y_train)
#displayResults(clf, "Orthogonal Matching Pursuit") 

## 1.1.9. Bayesian Regression
## 1.1.9.1. Bayesian Ridge Regression
#clf = linear_model.BayesianRidge().fit(X_train, y_train)
#displayResults(clf, "1.1.9.1. Bayesian Ridge Regression")

## 1.1.9.2. Automatic Relevance Determination - ARD
## Doesn't finish?
## clf = linear_model.ARDRegression().fit(X_train, y_train)
## displayResults(clf, "1.1.9.2. Automatic Relevance Determination - ARD")

## 1.1.10. Logistic regression
## Doesn't finish?
# clf = linear_model.LogisticRegression().fit(X_train, y_train)
# displayResults(clf, "1.1.10. Logistic regression")

## 1.1.11. Stochastic Gradient Descent - SGD / 
# clf = linear_model.SGDRegressor().fit(X_train, y_train) 
# displayResults(clf, "1.1.11. Stochastic Gradient Descent - SGD")

## 1.1.12. Perceptron
## ValueError("Unknown label type: %r" % ys)
#clf = linear_model.Perceptron().fit(X_train, y_train)
#displayResults(clf, "1.1.12. Perceptron")

## 1.1.13. Passive Aggressive Algorithms
#clf = linear_model.PassiveAggressiveRegressor().fit(X_train, y_train)
#displayResults(clf, "1.1.13. Passive Aggressive Algorithms")

## 1.1.14. Robustness regression: outliers and modeling errors
## 1.1.14.2. RANSAC: RANdom SAmple Consensus
#clf = linear_model.RANSACRegressor().fit(X_train, y_train)
#displayResults(clf, "1.1.14.2. RANSAC: RANdom SAmple Consensus")

## 1.1.14.3. Theil-Sen estimator: generalized-median-based estimator
#clf = linear_model.TheilSenRegressor().fit(X_train, y_train)
#displayResults(clf, "1.1.14.3. Theil-Sen estimator: generalized-median-based estimator")

## 1.1.15. Polynomial regression
## doesn't finish?
# clf = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', linear_model.LinearRegression(fit_intercept=False))])
# fit to an order-3 polynomial data
# clf = clf.fit(X_train, y_train)
# displayResults(clf, "1.1.15. Polynomial regression")


## 1.2. Linear and quadratic discriminant analysis
## Classifiers

# ## 1.3. Kernel ridge regression
# from sklearn.kernel_ridge import KernelRidge
# clf = KernelRidge().fit(X_train, y_train)
# displayResults(clf, "1.3. Kernel ridge regression")


## 1.4. Support Vector Machines
## 1.4.2. Regression
#clf = svm.SVR(kernel='rbf', C=1).fit(X_train, y_train)
#displayResults(clf, "1.4.2 SVR rbf")

# clf = svm.SVR(kernel='linear', C=1).fit(X_train, y_train)
# displayResults(clf, "1.4.2 SVR linear")

# clf = svm.SVR(kernel='sigmoid', C=1).fit(X_train, y_train)
# displayResults(clf, "1.4.2 SVR sigmoid")

#clf = svm.SVR(kernel='poly', C=1).fit(X_train, y_train)
#displayResults(clf, "1.4.2 SVR poly")


# ## NuSVR ##
# clf = svm.NuSVR(kernel='rbf', C=1).fit(X_train, y_train)
# displayResults(clf, "1.4.2 NuSVR rbf")

# clf = svm.NuSVR(kernel='linear', C=1).fit(X_train, y_train)
# displayResults(clf, "1.4.2 NuSVR linear")

# clf = svm.NuSVR(kernel='sigmoid', C=1).fit(X_train, y_train)
# displayResults(clf, "1.4.2 NuSVR sigmoid")

# clf = svm.NuSVR(kernel='poly', C=1).fit(X_train, y_train)
# displayResults(clf, "1.4.2 NuSVR poly")


## LinearSVR ##
# clf = svm.LinearSVR(C=1, loss="epsilon_insensitive").fit(X_train, y_train)
# displayResults(clf, "LinearSVR")


## 1.5. Stochastic Gradient Descent
## 1.5.2. Regression
## refer to 1.1.11. Stochastic Gradient Descent - SGD


## 1.6. Nearest Neighbors 
## 1.6.3. Nearest Neighbors Regression
# from sklearn import neighbors
# clf = neighbors.KNeighborsRegressor(weights = "uniform").fit(X_train, y_train)
# displayResults(clf, "1.6.3. KNeighborsRegressor uniform")

# clf = neighbors.KNeighborsRegressor(weights = "distance").fit(X_train, y_train)
# displayResults(clf, "1.6.3. KNeighborsRegressor distance")

# clf = neighbors.RadiusNeighborsRegressor(weights = "uniform").fit(X_train, y_train) # doesnt work?
# displayResults(clf, "1.6.3. RadiusNeighborsRegressor uniform")

# clf = neighbors.RadiusNeighborsRegressor(weights = "distance").fit(X_train, y_train) # doesnt work?
# displayResults(clf, "1.6.3. RadiusNeighborsRegressor distance")

#clf = neighbors.NearestCentroid().fit(X_train, y_train)
#displayResults(clf, "1.6.3. NearestCentroid")


## 1.7. Gaussian Processes
## too slow?
#from sklearn import gaussian_process
#clf = gaussian_process.GaussianProcess().fit(X_train, y_train)
#displayResults(clf, "1.7. Gaussian Processes")


## 1.8. Cross decomposition
#PLSRegression PLSCanonical, CCA and PLSSVD
#from sklearn import cross_decomposition
#clf = cross_decomposition.PLSRegression().fit(X_train, y_train)
#displayResults(clf, "1.8. Cross decomposition")

#clf = cross_decomposition.PLSCanonical().fit(X_train, y_train)
#displayResults(clf, "1.8. Cross decomposition")

#clf = cross_decomposition.CCA().fit(X_train, y_train)
#displayResults(clf, "1.8. Cross decomposition")


## 1.9. Naive Bayes (for classification?)
## NaiveBayes
# clf = naive_bayes.GaussianNB().fit(X_train, y_train)
# displayResults(clf, "1.9.1. GaussianNB")

# doesn't work for this dataset?
# clf = naive_bayes.MultinomialNB().fit(X_train, y_train)
# displayResults(clf, "1.9.2. MultinomialNB")

# doesn't work for this dataset?
# clf = naive_bayes.BernoulliNB().fit(X_train, y_train)
# displayResults(clf, "1.9.3. BernoulliNB")


## 1.10. Decision Trees
## Tree ##
# clf = tree.DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
# displayResults(clf, "1.10. DecisionTreeRegressor")

#clf = tree.ExtraTreeRegressor(random_state=0).fit(X_train, y_train)
#displayResults(clf, "1.10. ExtraTreeRegressor?")


## 1.11. Ensemble methods
#from sklearn import ensemble
# averaging methods
# 1.11.1. Bagging meta-estimator
#clf = ensemble.BaggingRegressor(neighbors.KNeighborsRegressor()).fit(X_train, y_train)
#displayResults(clf, "1.11.1. Bagging meta-estimator")

# 1.11.2.1. Random Forests
#clf = ensemble.RandomForestRegressor().fit(X_train, y_train)
#displayResults(clf, "1.11.2.1. Random Forests")

# 1.11.2.2. Extremely Randomized Trees
#clf = ensemble.ExtraTreesRegressor().fit(X_train, y_train)
#displayResults(clf, "1.11.2.2. Extremely Randomized Trees")

# boosting methods
# 1.11.3. AdaBoost
# clf = ensemble.AdaBoostRegressor().fit(X_train, y_train)
# displayResults(clf, "1.11.3. AdaBoost")

# 1.11.4. Gradient Tree Boosting
#clf = ensemble.GradientBoostingRegressor().fit(X_train, y_train)
#displayResults(clf, "1.11.4. Gradient Tree Boosting")


## 1.12. Multiclass and multilabel algorithms
# not regression 

## 1.13. Feature selection
# not about estimator

## 1.14. Semi-Supervised
# all samples have price data, so doesn't apply

## 1.15. Isotonic regression
# ValueError("X should be a 1d array")
#from sklearn.isotonic import IsotonicRegression
#clf = IsotonicRegression().fit(X_train, y_train)
#displayResults(clf, "1.14. Semi-Supervised")

## 1.16. Probability calibration




