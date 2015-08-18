import requests
import json
import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib
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


csvfile = 'HousePriceList_All_Train.csv'
# adding attributes did not make a difference
#csvfile = 'rentList_All_withAllAttr_trimmed.csv' 
#csvfile = 'rentList_E_EC_N_final.csv'
data = readCsvIntoPandasDataframe(csvfile)
X, y, X_scaler, y_scaler, header, imp, vec = preprocessData(data)


#print "header type:"
#print type(header).__name__
#print header
#X, y = preprocessDataWithoutScale(data)

joblib.dump(X_scaler, 'pickle-house/X_scaler.pkl') 
joblib.dump(y_scaler, 'pickle-house/y_scaler.pkl') 
joblib.dump(imp, 'pickle-house/Imputer.pkl') 
joblib.dump(vec, 'pickle-house/Vector.pkl') 

estimators = []

# K-Nearest Neighbors
estimators.append( {"name": "KNN", 
				   "model": neighbors.KNeighborsRegressor(
				   			weights = "uniform", n_neighbors = 5) } )

# Gradient Boosting Regressor
estimators.append( {"name": "GBR", 
				   "model": ensemble.GradientBoostingRegressor(
				   			max_features = 0.1, n_estimators = 2100, 
							max_depth = 6, min_samples_leaf = 1, learning_rate = 0.02) })

# Random Forest
estimators.append( {"name": "RF", 
				   "model": ensemble.RandomForestRegressor(
				   			max_features = 0.1, n_estimators = 512 ) })

# Support Vector Machine
estimators.append( {"name": "SVR", 
				   "model": svm.SVR(cache_size=1000, kernel='poly', 
				   			C = 1, gamma = 0.1, epsilon = 0.1, degree = 3) })

#kf = KFold(len(X), n_folds=10, shuffle=True, random_state=17) #3) #70) #48)
#kf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=3) #70) #48)


for estimator in estimators:
	print "== " + estimator['name'] + " =="
 	
 	model = estimator['model']
	start = time.time()
	model.fit(X, y)
	print("Fitting took %.2f seconds." % (time.time() - start) )
	joblib.dump(model, 'pickle-house/' + estimator['name'] + '.pkl') 


	if(estimator['name'] == 'GBR' or estimator['name'] == 'RF'):
		title = estimator['name']
		hdr, imp = plotImportances(model, header, title)
		#print hdr
		#print imp
		#o = np.concatenate((hdr, imp), axis=0)
		o = np.column_stack((hdr, imp))
		np.savetxt("pickle-house/FeatImp-" + estimator['name'] + ".csv",
					 o, delimiter=",", fmt="%s")
		#fout = open("pickle/FeatImp-" + estimator['name'] + "-" + str(i) + ".csv","a")
		#fout.write(hdr)
		#fout.write(imp)
		#fout.close()

