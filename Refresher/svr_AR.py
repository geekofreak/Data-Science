# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 19:36:30 2018

@author: arthy
"""
#DATA PREPROCESSING 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORT DATASET, CREATE A MATRIX OF FEATURES WITH INDEPENDENT AND A VECTOR OF DEPENDENT VARIABLE

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
y = np.reshape(y,(len(y),1))

#PREFROM FEAUTURE SCALING
from sklearn.preprocessing import StandardScaler
Scaled_X = StandardScaler()
x = Scaled_X.fit_transform(x)
Scaled_Y = StandardScaler()
y = Scaled_Y.fit_transform(y)
y = y.ravel()

#CREATE AND FIT A SVR MODEL
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor = regressor.fit(x,y)

#PREDICT THE SALARY USING SVR MODEL FOR LEVEL = 6.5
y_pred = Scaled_Y.inverse_transform(regressor.predict(Scaled_X.transform(np.array([[6.5]]))))

#VISUALIZE THE SVR MODEL
X_grid = np.arange(min(x),max(x),0.1)
X_grid = np.reshape(X_grid,(len(X_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
