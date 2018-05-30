# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:19:26 2018

@author: Mohit Uniyal
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values


"""from sklearn.cross_validation import train_test_split
X_train, X_test , Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)"""


from sklearn.linear_model import LinearRegression
lin_reg_1  = LinearRegression()
lin_reg_1.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,Y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

plt.scatter(X,Y, color ='Red')
plt.plot(X, lin_reg_1.predict(X), color ='Blue')
plt.title("Salary Vs. Level")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()



X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X,Y, color ='Red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color ='Blue')
plt.title("Salary Vs. Level")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()



lin_reg_1.predict(6.5)

lin_reg_2.predict(poly_reg.fit_transform(6.5))

