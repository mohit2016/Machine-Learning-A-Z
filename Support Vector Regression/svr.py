# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
Y = sc_y.fit_transform(Y)

from sklearn.svm import SVR
svr_regressor = SVR(kernel ="rbf")
svr_regressor.fit(X,Y)
print(svr_regressor.score)

y_pred = sc_y.inverse_transform(svr_regressor.predict(sc_x.transform(np.array([[6.5]]))))


X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color ="red")
plt.plot(X_grid, svr_regressor.predict(X_grid), color="blue")
plt.title("salary vs level")
plt.show()


