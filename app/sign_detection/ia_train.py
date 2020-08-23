#!/usr/bin/python3.6
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
 
batch_size_val = 10
steps_per_epoch_val = 2220
epochs_val = 10
imageDimesions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

def getData():
    path = "myData"
    images = []
    classNo = []
    myList = os.listdir(path)
    nbClass = len(myList)
    for x in range (0, len(myList)):
        myPicList = os.listdir(path + "/" + str(x))
        for y in myPicList:
            curImg = cv2.imread(path + "/" + str(x) + "/" + y)
            images.append(curImg)
            classNo.append(x)
    images = np.array(images)
    classNo = np.array(classNo)
    return images, classNo, nbClass

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def myModel(nbClass):
    nbFilters = 60
    Filter_1 = (5, 5)
    Filter_2 = (3, 3)
    size_of_pool = (2, 2)
    nbNodes = 500
    model = Sequential()
    model.add((Conv2D(nbFilters, Filter_1, input_shape = (imageDimesions[0], imageDimesions[1], 1), activation = 'relu')))
    model.add((Conv2D(nbFilters, Filter_1, activation = 'relu')))
    model.add(MaxPooling2D(pool_size = size_of_pool))
 
    model.add((Conv2D(nbFilters // 2, Filter_2, activation = 'relu')))
    model.add((Conv2D(nbFilters // 2, Filter_2, activation = 'relu')))
    model.add(MaxPooling2D(pool_size = size_of_pool))
    model.add(Dropout(0.5))
 
    model.add(Flatten())
    model.add(Dense(nbNodes, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nbClass, activation = 'softmax'))
    model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def error_handling(X_train, X_validation, X_test, Y_train, Y_validation, Y_test):
    print("Data Shapes")
    print("Train",end = "");print(X_train.shape,Y_train.shape)
    print("Validation",end = "");print(X_validation.shape,Y_validation.shape)
    print("Test",end = "");print(X_test.shape,Y_test.shape)
    if X_train.shape[0] != Y_train.shape[0]:
        print("The number of images in not equal to the number of labels in training set")
        return 84
    if X_validation.shape[0] != Y_validation.shape[0]:
        print("The number of images in not equal to the number of labels in validation set")
        return 84
    if X_test.shape[0] != Y_test.shape[0]:
        print("The number of images in not equal to the number of labels in test set")
        return 84
    if X_train.shape[1:] != (imageDimesions):
        print("The dimensions of the Training images are wrong")
        return 84
    if X_validation.shape[1:] != (imageDimesions):
        print("The dimensions of the Validation images are wrong")
        return 84
    if X_test.shape[1:] != (imageDimesions):
        print("The dimensions of the Test images are wrong")
        return 84
    return 0

def display_plot(history, model, X_test, Y_test):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training','validation'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training','validation'])
    plt.title('Acurracy')
    plt.xlabel('epoch')
    plt.show()
    score = model.evaluate(X_test, Y_test, verbose = 0)
    print('Test Score:', score[0])
    print('Test Accuracy:', score[1])

def main():
    images, classNo, nbClass = getData()
    X_train, X_test, Y_train, Y_test = train_test_split(images, classNo, test_size = testRatio)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size = validationRatio)
    if error_handling(X_train, X_validation, X_test, Y_train, Y_validation, Y_test) == 84:
        return 84

    data=pd.read_csv("labels.csv")

    X_train=np.array(list(map(preprocessing,X_train)))
    X_validation=np.array(list(map(preprocessing,X_validation)))
    X_test=np.array(list(map(preprocessing,X_test)))
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
    dataGen.fit(X_train)
    batches = dataGen.flow(X_train, Y_train, batch_size = 20)
    X_batch, Y_batch = next(batches)
    
    Y_train = to_categorical(Y_train, nbClass)
    Y_validation = to_categorical(Y_validation, nbClass)
    Y_test = to_categorical(Y_test, nbClass)

    model = myModel(nbClass)
    print(model.summary())
    history = model.fit(dataGen.flow(X_train, Y_train, batch_size = batch_size_val), steps_per_epoch = steps_per_epoch_val, epochs = epochs_val, validation_data = (X_validation, Y_validation), shuffle = 1)
    
    display_plot(history, model, X_test, Y_test)

    model.save('model.h5')
    cv2.waitKey(0)

if __name__ == "__main__":
    main()