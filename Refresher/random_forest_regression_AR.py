# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 21:57:21 2018

@author: arthy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORT THE DATASET
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#BUILD THE RANDOM FOREST MODEL
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(x,y)

#PREDICT THE SALARY FOR LEVEL = 6.5
y_pred = regressor.predict(6.5)

#VISUALIZE THE MODEL
x_grid = np.arange(min(x),max(x),0.01)
x_grid = np.reshape(x_grid,(len(x_grid),1))
plt.scatter(x,y,color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title('Random Forest Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()