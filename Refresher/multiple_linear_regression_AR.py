# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 22:47:26 2018

@author: arthy
"""

# Data Preprocessing Step

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:/DataScience/Machine Learning Course - Udemy/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encode categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:,3] = labelencoder_x.fit_transform(X[:,3])

onehotencoder_x = OneHotEncoder(categorical_features = [3])
X = onehotencoder_x.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling is automatically taken care off by the library

#Fit muliple linear regression on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Test the performance of the fitted model on the test set
y_predict = regressor.predict(X_test)

#Optimize the model with the best predictors using Backward Elimination method
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values= X, axis=1)

X_Opt = X[:,[0,1,2,3,4,5]]
regressor_OLS =sm.OLS(y, X_Opt).fit()
regressor_OLS.summary()

X_Opt = X[:,[0,1,3,4,5]]
regressor_OLS =sm.OLS(y, X_Opt).fit()
regressor_OLS.summary()

X_Opt = X[:,[0,3,4,5]]
regressor_OLS =sm.OLS(y, X_Opt).fit()
regressor_OLS.summary()

X_Opt = X[:,[0,3,5]]
regressor_OLS =sm.OLS(y, X_Opt).fit()
regressor_OLS.summary()

X_Opt = X[:,[0,3]]
regressor_OLS =sm.OLS(y, X_Opt).fit()
regressor_OLS.summary()


