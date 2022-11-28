# Author: John Prominski
# CS4342 - A3 - Q1
# 11/22/22

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
from sklearn.neighbors import KNeighborsRegressor

# load data
market_data = pd.read_csv("Weekly.csv")

# preview data
print(market_data.head())

# map Direction col values into numeric [down: 0, up: 1]
market_data.Direction = market_data.Direction.replace({'Down': 0, 'Up': 1})

# preview data types
print(market_data.info())

# -------------------------- PART A - VISUALIZE DATA --------------------------

# print correlation matrix for all market_data varibles
print("correlation matrix for : market_data\n", market_data.corr())

# plot scatter maxtrix for all market_data variables
pd.plotting.scatter_matrix(market_data)
#plt.show()

# ----------------------- PART B - LOGISTIC REGRESSION ------------------------

# Split data
# feature variables
X = market_data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
# dependant varible (Direction)
y = market_data[['Direction']]

print(y.head())

# perform logistic regression on full dataset
# print summary
X2 = sm.add_constant(X)
est = sm.Logit(y, X2)
est2 = est.fit()
print("----")
print(est2.summary())
print("----")

# ----------------------- PART C - COINFUSION MATRIX ------------------------

# perform logistic regression again (with sklearn..)
logreg = LogisticRegression(random_state=16)
res = logreg.fit(X, y)
y_pred = res.predict(X)

# get confusion matrix
c_matrix = metrics.confusion_matrix(y.values.reshape((-1,)), y_pred)
print(c_matrix)

# ----------------------- PART D - LOG REGRESSION FOR LAG2 ------------------------

# Split data (pre-2009 and post-2008)
data_train = market_data[market_data['Year'] < 2009]
data_test = market_data[market_data['Year'] >= 2009]

# feature train variables
X_train = data_train[['Lag2']]
# dependant train varibles (Direction)
y_train = data_train[['Direction']]

# feature test variables
X_test = data_test[['Lag2']]
# dependant test varibles (Direction)
y_test = data_test[['Direction']]


# perform logistic regression again (with sklearn..)
logreg = LogisticRegression(random_state=16)
res = logreg.fit(X_train, y_train)
y_pred = res.predict(X_test)

# get confusion matrix
print("-----Logistic regression:")
c_matrix = metrics.confusion_matrix(y_test.values.reshape((-1,)), y_pred)
print(c_matrix)

# ----------------------- PART E - LDA FOR LAG2 ------------------------

# scale data for LDA
sc= StandardScaler()
X_train_LDA = sc.fit_transform(X_train)
X_test_LDA = sc.transform(X_test)

# perform LDA
lda = LDA(n_components=1)
X_train_LDA = lda.fit_transform(X_train_LDA, y_train)
X_test_LDA = lda.transform(X_test_LDA)

classifier = RandomForestClassifier(max_depth=2, random_state =0)
classifier.fit(X_train_LDA, y_train)
y_pred = classifier.predict(X_test_LDA)

# get confusion matrix
print("-----LDA:")
c_matrix = metrics.confusion_matrix(y_test.values.reshape((-1,)), y_pred)
print(c_matrix)

# ----------------------- PART F - QDA FOR LAG2 ------------------------

model = QuadraticDiscriminantAnalysis()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# get confusion matrix
print("-----QDA:")
c_matrix = metrics.confusion_matrix(y_test.values.reshape((-1,)), y_pred)
print(c_matrix)

# ----------------------- PART G - KNN W/ K=1 FOR LAG2 ------------------------

regressor = KNeighborsRegressor(n_neighbors=1)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# get confusion matrix
print("-----KNN N=1:")
c_matrix = metrics.confusion_matrix(y_test.values.reshape((-1,)), y_pred)
print(c_matrix)
