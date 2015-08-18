import requests
import json
import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt

#from sklearn.externals import joblib
from mlhelper import readCsvIntoPandasDataframe, preprocessData, preprocessDataWithoutScale, plot_learning_curve
from sklearn import neighbors, cross_validation, metrics, ensemble, svm
from sklearn.cross_validation import KFold, StratifiedKFold

def plotImportances(clf, header, title):
	# Plot feature importance
	feature_importance = clf.feature_importances_
	# make importances relative to max importance
	feature_importance = 100.0 * (feature_importance / feature_importance.max())
	sorted_idx = np.argsort(feature_importance)
	pos = np.arange(sorted_idx.shape[0]) + .5
	plt.figure()
	plt.title(title)
	plt.subplot(1, 2, 2)
	plt.barh(pos, feature_importance[sorted_idx], align='center')
	plt.yticks(pos, header[sorted_idx]) # boston.feature_names[sorted_idx])
	plt.xlabel('Relative Importance')
	plt.title('Variable Importance')
	plt.show(block=False)

	return header[sorted_idx], feature_importance[sorted_idx]


csvfile = 'rentList_All_final.csv'
data = readCsvIntoPandasDataframe(csvfile)
X, y, X_scaler, y_scaler, header, imp, vec = preprocessData(data)

#joblib.dump(imp, 'pickle/Imputer.pkl') 
#joblib.dump(vec, 'pickle/Vector.pkl') 

#kf = KFold(len(X), n_folds=10, shuffle=True, random_state=17) #3) #70) #48)
kf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=3) #70) #48)

std_score = []
i = 1
for train_index, test_index in kf:
	print "== iteration " + str(i) + " =="
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	#print X_train[1]
	#print y_test[1]
 	
	std = np.std(y_test)
	print "std:", std

	std_score.append(std)
	i = i + 1

print "scores:"
print std_score
np.savetxt("standardDeviation-AllData.csv",	std_score, delimiter=",", fmt="%s")

#print estimator['name'] + " average:"
#print np.mean(validation_scores)


