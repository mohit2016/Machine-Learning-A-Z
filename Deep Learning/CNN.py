# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:26:19 2018

@author: Mohit Uniyal
"""

#Importing Libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing Neural Network
classifier = Sequential()

#Adding Convolutional Layer
classifier.add(Convolution2D(filters=32,kernel_size =[3,3], input_shape = (64,64,3), activation = 'relu'))
#Adding max pooling layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding 2nd COnvolutional Layer
classifier.add(Convolution2D(filters=32,kernel_size=[3,3], activation="relu"))
#Adding 2nd max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening Layers
classifier.add(Flatten())

#Full Conection layer
classifier.add(Dense(units =128, activation = 'relu'))
#output layer
classifier.add(Dense(units =1 , activation = 'sigmoid'))
#compile
classifier.compile(optimizer = 'adam', loss= "binary_crossentropy", metrics  = ['accuracy'])


#Image Augmentation
from keras.preprocessing.image import ImageDataGenerator

#Training Data Preprocessing
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

from PIL import Image

#Fit the training data and also check on test_set
classifier.fit_generator(
        generator=training_set,
        steps_per_epoch=250,
        epochs=25,
        validation_data=test_set,
        validation_steps=62)

import numpy as np
from keras.preprocessing import image

#Image preprocess for single prediction
test_img = image.load_img("dataset/image.jpeg",target_size=(64,64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img,axis=0)
result = classifier.predict(test_img)
training_set.class_indices

prediction =""
if result[0][0] == 1:
    prediction="dog"
else:
    prediction="cat"