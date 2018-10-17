# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 19:42:20 2018

@author: arthy
"""
#DATA PREPROCESSING

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

"""
Simple Linear Regression model from the Scikit library takes care of feature scaling 
internally. Hence explicit feature scaling of the independent variable is not needed
"""

"""
Import Linear Regression model from Scikit library and fit the simple linear regression 
to the training set
"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

""""
Predict Y variable for the test set using the model built in the previous step
""""
y_predict = regressor.predict(X_test)

"""
Visualize the training set data and results
"""
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Training Set')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

"""
Visualize the test set data and results
"""
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title('Testing Set')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()
