# Author: John Prominski
# CS4342 - A3 - Q2
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
auto_data = pd.read_csv("Auto.csv")

# preview data
print(auto_data.head())

# -------------------------- PART A - ADD MPG01 COL TO DF --------------------------

# find median for mpg
print(auto_data.describe())

median = 22.75

# function defining mpg01
# return 1 if mpg is above median mpg, otherwise 0
def mpg01 (row):
    if row['mpg'] > median:
        return 1
    else:
        return 0

# add new column based on mpg01 rules
auto_data['mpg01'] = auto_data.apply(lambda row: mpg01(row), axis=1)

# preview data with added mpg01 column
print(auto_data.head())

# -------------------------- PART B - VISUALIZE DATA --------------------------

# print correlation matrix for all market_data varibles
print("correlation matrix for : auto_data\n", auto_data.corr())

# plot scatter maxtrix for all market_data variables
print("scatter matrix for : auto_data:")
pd.plotting.scatter_matrix(auto_data)
#plt.show()

# -------------------------- PART C - SPLIT DATA --------------------------

# Split data into feature and response
# feature variables (highest correlation variables)
X = auto_data[['cylinders', 'displacement', 'weight']]
# dependant varible (mpg01)
y = auto_data[['mpg01']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# -------------------------- PART D - LDA  --------------------------

# perform LDA with cylinders, displacement, weight
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

# -------------------------- PART E - QDA  --------------------------

# fit model
model = QuadraticDiscriminantAnalysis()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# get confusion matrix
print("----- Confusion Matrix for QDA:")
c_matrix = metrics.confusion_matrix(y_test.values.reshape((-1,)), y_pred)
print(c_matrix)

# print test error 
print("test error of QDA:")
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
