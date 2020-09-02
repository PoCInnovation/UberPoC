#!/usr/bin/env python3

from pyimagesearch import config
from imutils import paths
import numpy as np
import pickle
import os
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.models import Model
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator



class RCNN():
    def __init__(self):
        self.init_lr = 1e-4
        self.epochs = 5
        self.bs = 2
        self.baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
        self.model = None
        self.aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )

#        self.build_model()

    def load_dataset(self):
        imagePaths = list(paths.list_images(config.BASE_PATH))


    def build_model(self):
        headModel = self.baseModel.output
        headModel = AveragePooling2D(pool_size=(7,7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(len(config.LABELS), activation="softmax")(headModel)

        self.model = Model(inputs=self.baseModel.input, outputs=headModel)
        for layer in self.baseModel.layers:
            layer.trainable = False
        return self

    def summary(self):
        self.model.summary()

    def compile(self):
        print("[+] Model is compiling...")
        opt = Adam(lr=self.init_lr)
        self.model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return self

    def train(self):
        print("[+] Model is training...")
        H = self.model.fit(
            self.aug.flow(...),
            steps_per_epoch=...,
            validation_data=(),
            validation_steps=(),
            epochs=self.epochs
        )
        return self

myRcnn = RCNN()
myRcnn.load_dataset()