import json
import csv
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as DV
import pandas as pd

csvfile = 'rentList_E_WS_BoroWard-trimmed.csv'

train_file = 'rentList_E_reduced_train.csv'
test_file = 'rentList_E_reduced_test.csv'

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

train = train[train.walkscore.notnull()]
test = test[test.walkscore.notnull()]


numeric_cols = [ 'num_bathrooms', 'num_bedrooms', 'num_floors', 'num_recepts', 'latitude', 'longitude', 'walkscore' ]
x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

# scale to <0,1>

max_train = np.amax( x_num_train, 0 )
max_test = np.amax( x_num_test, 0 )	

x_num_train = x_num_train / max_train
x_num_test = x_num_test / max_train		# scale test by max_train


# y
y_train = train.price
y_test = test.price


# categorical

cat_train = train.drop( numeric_cols + [ 'price'], axis = 1 )
cat_test = test.drop( numeric_cols + [ 'price'], axis = 1 )

cat_train.fillna( 'NA', inplace = True )
cat_test.fillna( 'NA', inplace = True )

x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()

print x_cat_test

# vectorize

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )

print vec_x_cat_test

# complete x

x_train = np.hstack(( x_num_train, vec_x_cat_train ))
x_test = np.hstack(( x_num_test, vec_x_cat_test ))

svc = svm.SVC(kernel='linear')#(kernel='linear')
svc.fit(x_train, y_train)    
print svc

print "\nresult of prediction:" 
print svc.predict(x_test);

print "\nactual data:"
print y_test

# test = pd.read_csv( test_file )


# fin = open(csvfile, 'r')
# reader = csv.DictReader(fin)

# data = []
# num_row = 0

# for row in reader:
# 	if(num_row >= 100):
# 		break

# 	if row["walkscore"] is not None:
# 		# data.append([row["num_bathrooms"],\
# 		# 			row["num_bedrooms"],\
# 		# 			row["num_floors"],\
# 		# 			row["num_recepts"],\
# 		# 			#row["outcode"],\
# 		# 			row["latitude"],\
# 		# 			row["longitude"],\
# 		# 			row["property_type"],\
# 		# 			row["walkscore"],\
# 		# 			row["price"]])
# 		data.append(row)
# 		num_row += 1

# print "the number of row is " + str(num_row)

# print data[0]
# print data[1]

# #print(data)
# dataset = np.array(data)#loadtxt(data, delimiter=",")
# print(dataset.shape)
# # separate the data from the target attributes
# X = dataset[:,0:7]
# Y = dataset[:,8]

# #np.unique(iris_y)
# print X[0]
# print X[1]


# # Standardize data
# scaler = preprocessing.StandardScaler().fit(X)
# scaler.transform(X)                               

# print X[0]
# print X[1]


# np.random.seed(0)
# indices = np.random.permutation(len(X))
# X_train = X[indices[:-10]]
# Y_train = Y[indices[:-10]]
# X_test  = X[indices[-10:]]
# Y_test  = Y[indices[-10:]]

# svc = svm.SVC(kernel='rbf')#(kernel='linear')
# svc.fit(X_train, Y_train)    
# print svc

# print "\nresult of prediction:" 
# print svc.predict(X_test);

# print "\nactual data:"
# print Y_test


