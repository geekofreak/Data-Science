# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:04:54 2018

@author: arthy
"""
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Build Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor_linear = LinearRegression()
regressor_linear.fit(X,y)

#Build Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
regressor_poly = PolynomialFeatures(degree = 4)

#Create a polynomial version of X variable
X_poly  = regressor_poly.fit_transform(X)

regressor_linear_poly = LinearRegression()
regressor_linear_poly.fit(X_poly,y)

#Visualizing Linear Regression Model
plt.scatter (X,y,color = 'red')
plt.plot(X,regressor_linear.predict(X), color = 'blue')
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#Visualizing Polynomial Regression 
X_grid = np.arange(min(X),max(X),0.05)
X_grid = np.reshape(X_grid, (len(X_grid),1))
plt.scatter (X,y,color = 'red')
plt.plot(X_grid,regressor_linear_poly.predict(regressor_poly.fit_transform(X_grid)), color = 'green')
plt.title("Polynomial Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#Predict the salary for the employee at level 6.5 using linear regression
regressor_linear.predict(6.5)

#Predict the salary for the employee at level 6.5 using polynomial regression
regressor_linear_poly.predict(regressor_poly.fit_transform(6.5))
