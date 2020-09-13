from .pyimagesearch.nms import non_max_suppression
from .pyimagesearch import config

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

model = load_model(f"{os.getcwd()}/app/sign_detection/models/sign_detector.optimized.h5")
lb = pickle.loads(open(f"{os.getcwd()}/app/sign_detection/label_encoder.pickle", "rb").read())

def predict(img):
    img = imutils.resize(img, width=500)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    proposals = []
    boxes = []

    for (x, y, w, h) in rects[:config.MAX_PROPOSALS_INFER]:
        # extract the region from the input image, convert it from BGR to
        # RGB channel ordering, and then resize it to the required input
        # dimensions of our trained CNN
        roi = img[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, config.INPUT_DIMS,
                         interpolation=cv2.INTER_CUBIC)
        # further preprocess by the ROI
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        # update our proposals and bounding boxes lists
        proposals.append(roi)
        boxes.append((x, y, x + w, y + h))

    proposals = np.array(proposals, dtype="float32")
    boxes = np.array(boxes, dtype="int32")
    proba = model.predict(proposals)

    # NMS (Non max suppresion)
    labels = lb.classes_[np.argmax(proba, axis=1)]
    idxs = np.where(labels == "stop_sign")[0]

    # use the indexes to extract all bounding boxes and associated class
    # label probabilities associated with the "raccoon" class
    boxes = boxes[idxs]
    proba = proba[idxs][:, 1]

    # further filter indexes by enforcing a minimum prediction
    # probability be met
    idxs = np.where(proba >= config.MIN_PROBA)
    boxes = boxes[idxs]
    proba = proba[idxs]
    # run non-maxima suppression on the bounding boxes
    boxIdxs = non_max_suppression(boxes, proba)

    for i in boxIdxs:
        # draw the bounding box, label, and probability on the image
        (startX, startY, endX, endY) = boxes[i]
        cv2.rectangle(img, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        text = "Stop Sign: {:.2f}%".format(proba[i] * 100)
        cv2.putText(img, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    return img