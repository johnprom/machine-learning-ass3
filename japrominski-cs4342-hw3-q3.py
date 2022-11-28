# Author: John Prominski
# CS4342 - A3 - Q3
# 11/22/22

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

# load data
boston_data = pd.read_csv("Boston.csv")

# preview data
print(boston_data.head())

# find median for mpg
print(boston_data.describe())

median = 0.256510

# return 1 if crim is above median crim, otherwise 0
def med_crim (row):
    if row['crim'] > median:
        return 1
    else:
        return 0

# add new column based on med_crim rules
boston_data['med_crim'] = boston_data.apply(lambda row: med_crim(row), axis=1)

# preview data with added med_crim column
print(boston_data.head())

# print correlation matrix for all boston_data varibles
print("correlation matrix for : auto_data\n", boston_data.corr())

# Split data into feature and response
# feature variables (highest correlation variables)
X = boston_data[['nox', 'dis', 'age', 'rm', 'rad', 'indus']]
# dependant varible (crim)
y = boston_data[['med_crim']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# -------------------------- LDA  --------------------------

# perform LDA with nox, dis, age, rm, rad, indus
model = RandomForestClassifier(max_depth=2, random_state =0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# get confusion matrix
print("----- Confusion Matrix for LDA:")
c_matrix = metrics.confusion_matrix(y_test.values.reshape((-1,)), y_pred)
print(c_matrix)

# print test error 
print("test error of LDA:")
print(accuracy_score(y_test, y_pred))

# -------------------------- PART F - Logistic Regression  --------------------------

# perform logistic regression again (with sklearn..)
logreg = LogisticRegression(random_state=16)
res = logreg.fit(X_train, y_train)
y_pred = res.predict(X_test)

# get confusion matrix
print("----- Confusion Matrix for Logistic Regression:")
c_matrix = metrics.confusion_matrix(y_test.values.reshape((-1,)), y_pred)
print(c_matrix)

# print test error 
print("test error of Logistic Regression:")
print(accuracy_score(y_test, y_pred))

# -------------------------- PART G - KNN (muliple k vals)  --------------------------

# fit model
knn_model = KNeighborsRegressor(n_neighbors=1)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

# get confusion matrix
print("-----KNN N=1:")
c_matrix = metrics.confusion_matrix(y_test.values.reshape((-1,)), y_pred)
print(c_matrix)

# print test error 
print("test error of KNN:")
print(accuracy_score(y_test, y_pred))
