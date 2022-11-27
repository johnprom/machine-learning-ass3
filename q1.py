# Author: John Prominski
# CS4342 - A3 - Q1
# 11/22/22

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm

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

# perform linear regression on full dataset
#logreg = LogisticRegression(random_state=16)
#res = logreg.fit(X, y)

log_reg = sm.Logit(y, X).fit()

print("----")
log_reg.summary()
print("----")

y_pred = log_reg.predict(X)

# get confusion matrix
c_matrix = metrics.confusion_matrix(y.values.reshape((-1,)), y_pred)
print(c_matrix)





#perform a logistic regression with Direction as the response and the five lag 
#variables plus Volume as predictors