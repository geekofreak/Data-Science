# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:50:06 2018

@author: arthy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORT DATASET
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#BUILD THE DECISION TREE MODEL AND FIT IT TO X
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)

#PREDICT USING THE MODEL BUILT ABOVE
y_pred = regressor.predict(6.5)

#VISUALIZE THE RESULTS
x_grid = np.arange(min(x),max(x),0.01)
x_grid = np.reshape(x_grid,(len(x_grid),1))
plt.scatter(x,y,color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title ('Decision Tree Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()