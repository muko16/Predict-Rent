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

csvfile = 'newRentList_validation_trimmed.csv'
# adding attributes did not make a difference
#csvfile = 'rentList_All_withAllAttr_trimmed.csv' 
#csvfile = 'rentList_E_EC_N_final.csv'
data = readCsvIntoPandasDataframe(csvfile)

X_scaler = joblib.load('pickle/X_scaler.pkl') 
y_scaler = joblib.load('pickle/y_scaler.pkl') 
imp = joblib.load('pickle/Imputer.pkl') 
vec = joblib.load('pickle/Vector.pkl') 

X, y = preprocessDataWithScaler(data, X_scaler, y_scaler, imp, vec)
print np.std(y)

# estimators = []

# # K-Nearest Neighbors
# estimators.append( {"name": "KNN" } )

# # Gradient Boosting Regressor
# estimators.append( {"name": "GBR" } ) 

# # Random Forest
# estimators.append( {"name": "RF" } ) 

# # Support Vector Machine
# estimators.append( {"name": "SVR" } )

# #kf = KFold(len(X), n_folds=10, shuffle=True, random_state=17) #3) #70) #48)
# #kf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=3) #70) #48)

# validation_scores_all = []
# for estimator in estimators:
# 	validation_scores = []
# 	#i = 1
# 	#for train_index, test_index in kf:
# 	for i in range(1, 11):
# 		print "== " + estimator['name'] + ": iteration " + str(i) + " =="
# 		#X_train, X_test = X[train_index], X[test_index]
# 		#y_train, y_test = y[train_index], y[test_index]
# 		#print X_train[1]
# 		#print y_test[1]
	 	
# 	 	model = joblib.load('pickle/' + estimator['name'] + '-' + str(i) + '.pkl') 
		
# 		start = time.time()
# 		y_predicted = model.predict(X)
# 		print("Predicting took %.2f seconds." % (time.time() - start) )

# 		sme = metrics.mean_squared_error(y_predicted, y)
# 		print "SME:", sme

# 		validation_scores.append(sme)

# 		i += 1

# 	print estimator['name'] + " scores:"
# 	print validation_scores
# 	print estimator['name'] + " average:"
# 	print np.mean(validation_scores)

# 	validation_scores_all.append(validation_scores)


# scores = np.asarray(validation_scores_all)
# names = np.asarray(['KNN', 'GBR', 'RF', 'SVR'])
# score = np.column_stack((names, scores))
# np.savetxt("pickle/validScores-newData.csv",	score, delimiter=",", fmt="%s")

# input("Press Enter to continue...")


