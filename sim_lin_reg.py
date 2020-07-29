# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 19:12:58 2020

@author: dheeraj_reddy_peram
"""
#Importing required libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset is read using pandas library
dataset = pd.read_csv('Salary_Data.csv')

#Splitting Dataset into dependent and Independent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Preparing training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Importing linear regression class from the sklearn library
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#Fitting the Data
regressor.fit(X_train, y_train )

#Predicting the values
y_pred = regressor.predict(X_test)

#Plotting the training set values onto a graph using matplotlib library
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title('Salary vs Exp(training_set')
plt.xlabel('yrs of Exp')
plt.ylabel('Salary')
plt.show()

#Plotting test set values onto a graph

plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title('Salary vs Exp(test_set')
plt.xlabel('yrs of Exp')
plt.ylabel('Salary')
plt.show()