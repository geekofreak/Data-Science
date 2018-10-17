# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#IMPORT NECESSARY LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORT DATASET
dataset = pd.read_csv('Data.csv')

#CREATE A MATRIX OF FEATURES
x = dataset.iloc[:,:-1].values

#CREATE A VECTOR OF DEPENDENT VARIABLE
y = dataset.iloc[:,3].values

#IMPORT IMPUTER CLASS FROM PREPROCESSING LIBRARY IN SCKIT LEARN
from sklearn.preprocessing import Imputer

#CREATE IMPUTER OBJECT
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)

#FIT THE IMPUTER OBJECT TO THE MATRIX OF FEATURES ON COLUMNS WITH MISSING VALUES
imputer = imputer.fit(x[:,1:3])

#TRANSFORM THE MATRIX OF FEATURES TO HANDLE MISSING VALUES
x[:,1:3] = imputer.transform(x[:,1:3])

#IMPORT LABEL ENCODER AND ONE HOT ENCODER CLASSES TO ENCODE CATEGORICAL VARIABLES
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])

onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#SPLIT THE DATA SET INTO TRAINING AND TESTING SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 0)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

