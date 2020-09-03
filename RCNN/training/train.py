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
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


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
        self.trainX, self.trainY = None, None
        self.testX, self.testY = None, None
        self.H = None
        self.lb = None
        self.build_model()

    def load_dataset(self):
        imagePaths = list(paths.list_images(config.BASE_PATH))
        data = []
        labels = []

        for imagePath in imagePaths:
            label = imagePath.split(os.path.sep)[-2]
            image = load_img(imagePath, target_size=config.INPUT_DIMS)
            image = img_to_array(image)
            image = preprocess_input(image)

            data.append(image)
            labels.append(label)
        data = np.array(data, dtype="float32")
        labels = np.array(labels)
        self.lb = LabelBinarizer()
        labels = self.lb.fit_transform(labels)
        labels = to_categorical(labels)
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(
            data, labels,
            test_size=0.20,
            stratify=labels,
            random_state=42
        )
        return self

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
        self.H = self.model.fit(
            self.aug.flow(self.trainX, self.trainY, batch_size=self.bs),
            steps_per_epoch=len(self.trainX) // self.bs,
            validation_data=(self.testX, self.testY),
            validation_steps=len(self.testX) // self.bs,
            epochs=self.epochs
        )
        return self

    def evaluate(self):
        print("[INFO] evaluating network...")
        predIdxs = self.model.predict(self.testX, batch_size=self.bs)

        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        predIdxs = np.argmax(predIdxs, axis=1)

        # show a nicely formatted classification report
        print(classification_report(self.testY.argmax(axis=1), predIdxs,
                                    target_names=self.lb.classes_))

        # serialize the model to disk
        print("[+] saving mask detector model...")
        self.model.save(config.MODEL_PATH, save_format="h5")

        # serialize the label encoder to disk
        print("[+] saving label encoder...")
        f = open(config.ENCODER_PATH, "wb")
        f.write(pickle.dumps(self.lb))
        f.close()

        # plot the training loss and accuracy
        N = self.epochs
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), self.H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig('test.png')


myRcnn = RCNN()
myRcnn.load_dataset()
myRcnn.compile()
myRcnn.train()
myRcnn.evaluate()