# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 21:28:21 2018

@author: Mohit Uniyal
"""
#Importing Libraries
import numpy as np
import pandas as pd

#Reading Dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X= dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#Take care of categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#split the data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

#PART-1 Creating ANN

#define nerual network
classifier = Sequential()

#add first layer and input layer
classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu", input_dim=11))

#add second hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform",activation="relu"))

#add ouput layer
classifier.add(Dense(units=1, kernel_initializer="uniform",activation="sigmoid"))

#compile ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])

#fitting ANN
classifier.fit(X_train,y_train,batch_size=10,epochs=100)



#PART-2 K-Fold Cross Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    cl = Sequential()
    cl.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_dim=11))
    cl.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
    cl.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
    cl.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    return cl
    
classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print(accuracies.mean())
print(accuracies.std())


#PART-3 TUNING Parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
def build_classifier(optimizer):
    cl = Sequential()
    cl.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_dim=11))
    cl.add(Dropout(rate=0.1))
    cl.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
    cl.add(Dropout(rate=0.1))
    cl.add(Dense(units=1,kernel_initializer="uniform",activation="relu"))
    cl.compile(optimizer=optimizer,loss="binary_crossentropy", metrics=["accuracy"])
    return cl

classifier = KerasClassifier(build_fn=build_classifier)

parameters= {
        'batch_size': [20,25,30],
        'epochs':[100,200,300],
        'optimizer':["adam","rmsprop"]
        }
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring="accuracy",cv=10)
grid_search =grid_search.fit(X_train,y_train)
print(grid_search.best_score_)
best_params = grid_search.best_params_


#predict values
y_pred = classifier.predict(X_test)
y_pred = y_pred>0.5

#check confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
