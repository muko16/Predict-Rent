import requests
import json
import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from mlhelper import readCsvIntoPandasDataframe, preprocessData, \
				preprocessDataWithoutScale, plot_learning_curve, preprocessDataWithScaler
from sklearn import neighbors, cross_validation, metrics, ensemble, svm
from sklearn.cross_validation import KFold, StratifiedKFold

csvfile = 'HousePriceList_All_Test.csv'
# adding attributes did not make a difference
#csvfile = 'rentList_All_withAllAttr_trimmed.csv' 
#csvfile = 'rentList_E_EC_N_final.csv'
data = readCsvIntoPandasDataframe(csvfile)

X_scaler = joblib.load('pickle-house/X_scaler.pkl') 
y_scaler = joblib.load('pickle-house/y_scaler.pkl') 
imp = joblib.load('pickle-house/Imputer.pkl') 
vec = joblib.load('pickle-house/Vector.pkl') 

X, y = preprocessDataWithScaler(data, X_scaler, y_scaler, imp, vec)
print np.std(y)

estimators = []

# K-Nearest Neighbors
estimators.append( {"name": "KNN" } )

# Gradient Boosting Regressor
estimators.append( {"name": "GBR" } ) 

# Random Forest
estimators.append( {"name": "RF" } ) 

# Support Vector Machine
estimators.append( {"name": "SVR" } )


validation_scores_all = []
for estimator in estimators:

	print "== " + estimator['name'] + " =="
	#X_train, X_test = X[train_index], X[test_index]
	#y_train, y_test = y[train_index], y[test_index]
	#print X_train[1]
	#print y_test[1]
 	
 	model = joblib.load('pickle-house/' + estimator['name'] + '.pkl') 
	
	start = time.time()
	y_predicted = model.predict(X)
	print("Predicting took %.2f seconds." % (time.time() - start) )

	mse = metrics.mean_squared_error(y_predicted, y)
	print "MSE:", mse

	validation_scores_all.append(mse)

	result = np.column_stack(( y_scaler.inverse_transform(y), y_scaler.inverse_transform(y_predicted) ))
	np.savetxt('pickle-house/' + estimator['name'] + '-result.csv', result, delimiter=",", fmt="%s")

scores = np.asarray(validation_scores_all)
names = np.asarray(['KNN', 'GBR', 'RF', 'SVR'])
score = np.column_stack((names, scores))
np.savetxt("pickle-house/validScores-newData.csv",	score, delimiter=",", fmt="%s")

input("Press Enter to continue...")


