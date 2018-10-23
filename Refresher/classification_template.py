# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:37:52 2018

@author: arthy
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:46:16 2018

@author: arthy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORT THE DATASET TO BE ANALYZED
dataset = pd.read_csv('Social_Network_Ads.csv')

#CREATE THE MATRIX OF FEATURES AND Y VECTOR
x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

#SPLIT THE DATASET INTO TRAINING AND TESTING DATASETS
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)

#SCALE THE FEATURES TO AVOID THE MODEL SKEWING TOWARDS THE VARIABLE WITH HIGHER VALUES
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#BUILD AND FIT THE CLASSIFIER


#PREDICT THE TEST SET RESULTS USING THE LOGISTIC MODEL THAT WAS FITTED EARLIER
y_test_predict = classifier.predict(x_test)

#EVALUATE THE LOGISTIC REGRESSION CLASSIFIER USING CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_test_predict)

#VISUALIZING THE TRAINING SET RESULTS
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# VISUALIZING THE TEST SET RESULTS
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()