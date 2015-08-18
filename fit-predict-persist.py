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


csvfile = 'rentList_All_final.csv'
# adding attributes did not make a difference
#csvfile = 'rentList_All_withAllAttr_trimmed.csv' 
#csvfile = 'rentList_E_EC_N_final.csv'
data = readCsvIntoPandasDataframe(csvfile)
X, y, X_scaler, y_scaler, header = preprocessData(data)

#print "header type:"
#print type(header).__name__
#print header
#X, y = preprocessDataWithoutScale(data)

joblib.dump(X_scaler, 'pickle/X_scaler.pkl') 
joblib.dump(y_scaler, 'pickle/y_scaler.pkl') 


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
kf = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=3) #70) #48)


validation_scores_all = []
for estimator in estimators:
	validation_scores = []
	i = 1
	for train_index, test_index in kf:
		print "== " + estimator['name'] + ": iteration " + str(i) + " =="
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		#print X_train[1]
		#print y_test[1]
	 	
	 	model = estimator['model']
		start = time.time()
		model.fit(X_train, y_train)
		print("Fitting took %.2f seconds." % (time.time() - start) )
		joblib.dump(model, 'pickle/' + estimator['name'] + '-' + str(i) + '.pkl') 

		start = time.time()
		y_predicted = model.predict(X_test)
		print("Predicting took %.2f seconds." % (time.time() - start) )

		sme = metrics.mean_squared_error(y_predicted, y_test)
		print "SME:", sme

		validation_scores.append(sme)

		if(estimator['name'] == 'GBR' or estimator['name'] == 'RF'):
			title = estimator['name'] + "-" + str(i)
			hdr, imp = plotImportances(model, header, title)
			#print hdr
			#print imp
			#o = np.concatenate((hdr, imp), axis=0)
			o = np.column_stack((hdr, imp))
			np.savetxt("pickle/FeatImp-" + estimator['name'] + "-" + str(i) + ".csv",
						 o, delimiter=",", fmt="%s")
			#fout = open("pickle/FeatImp-" + estimator['name'] + "-" + str(i) + ".csv","a")
			#fout.write(hdr)
			#fout.write(imp)
			#fout.close()

		i += 1

	print estimator['name'] + " scores:"
	print validation_scores
	print estimator['name'] + " average:"
	print np.mean(validation_scores)

	validation_scores_all.append(validation_scores)


scores = np.asarray(validation_scores_all)
names = np.asarray(['KNN', 'GBR', 'RF', 'SVR'])
score = np.column_stack((names, scores))
np.savetxt("pickle/scores.csv",	score, delimiter=",", fmt="%s")

input("Press Enter to continue...")

#title = "learning curve (k = 5, cv = 10)"
#plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.01), cv=kf, n_jobs=-1)

# import matplotlib.pyplot as plt

# ylim=(0.0, 1.01)

# plt.figure()
# plt.title(title)
# if ylim is not None:
#     plt.ylim(*ylim)
# plt.xlabel("# of stages")
# plt.ylabel("Score (MSE)")

# #train_scores_mean = np.mean(train_scores*-1, axis=1)
# #train_scores_std = np.std(train_scores*-1, axis=1)
# #test_scores_mean = np.mean(test_scores*-1, axis=1)
# #test_scores_std = np.std(test_scores*-1, axis=1)
# plt.grid()

# train_scores_mean = [ 0.00803554, 0.05226865, 0.1152227,  0.10331965 ,0.08469347,  0.17073652, 0.15071575,  0.20203074 , 0.19056794,  0.30100884]
# test_scores_mean = [ 0.94204332 , 0.89289978 , 0.74689466 , 0.75548599 , 0.74747293 , 0.61745336, 0.61973232,  0.57455973,  0.57165043 , 0.47745441]
# print "train_scores:"
# print train_scores_mean
# print "test_scores:"
# print test_scores_mean

# param_range = np.linspace(.1, 1.0, 10) * len(X)
# print param_range

#param_range = np.linspace(1, 6, 6, dtype = 'int').tolist() 


#plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
#plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation score")

# plt.legend(loc="best")

# plt.show()











# estimator.fit(X, y)

# #s = pickle.dumps(estimator)
# #print s

# from sklearn.externals import joblib

# joblib.dump(estimator, 'pickle/filename.pkl') 
# clf2 = joblib.load('pickle/filename.pkl') 

# #print clf2

# #s = pickle.dumps(clf2)
# #print s

# #clf2 = pickle.loads(s)
# print estimator.predict(X[0])
# print clf2.predict(X[0])
# print y[0]

# joblib.dump(scaler, 'pickle/scaler.pkl') 
# scaler2 = joblib.load('pickle/scaler.pkl') 


# print scaler2.inverse_transform( clf2.predict(X[0]) )
# print scaler2.inverse_transform( y[0] )