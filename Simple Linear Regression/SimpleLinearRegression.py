# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:19:11 2018

@author: Mohit Uniyal
"""

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Spliting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 1/3 , random_state = 0)

#Fitiing Simple Linear Regression to the Data set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the values of X_test
Y_pred = regressor.predict(X_test)

#Graph of Training set
plt.scatter(X_train, Y_train, color ="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary Vs Experience (Training Dataset)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#Graph of Test set
plt.scatter(X_test, Y_test, color ="red")
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title("Salary Vs Experience (Test Dataset)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
