from visualizer import visualizer

import pyglet
from pyglet.window import key
from pyglet.gl import *

import numpy as np
import cv2
import sys

from keras.models import load_model

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

def getClassName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

class SignMedia:
    def __init__(self, target):
        if target == "cam":
            self.media = cv2.VideoCapture(0)
            if not self.media.isOpened():
                raise FileNotFoundError
        else:
            self.media = cv2.imread(target)
            if self.media is None:
                raise FileNotFoundError
        self.target = target

    def getImageWithSignDetection(self):
        if self.target == "cam":
            if self.media.isOpened() is False:
                sys.exit(1)
            ret, frame = self.media.read()
            if ret is False:
                return None
        else:
            frame = self.media.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        model=load_model('model.h5')
        font = cv2.FONT_HERSHEY_SIMPLEX
        threshold = 0.75
        ### You need to apply the SignDetection AI on this part and return frame with the predict
        img = np.asarray(frame)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)
        cv2.putText(frame, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue =np.amax(predictions)
        if probabilityValue > threshold:
            cv2.putText(frame,str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    def __str__(self):
        return self.target



    def isOpened(self):
        return self.media.isOpened()

class SignVisualizer(visualizer.Visualizer):

    def __init__(self, target="cam"):
        super(SignVisualizer, self).__init__(duckietown=False)
        try:
            self.signMedia = SignMedia(target)
        except FileNotFoundError:
            print("Couldn't open the signMedia")
            sys.exit(1)

    def update(self, dt):
        frame = self.signMedia.getImageWithSignDetection()
        self.show(frame)

    def show(self, obs):
        result = obs
        super().show(result)

    def on_key_press(self, symbol, identifiers):
        return

    def run(self):
        fps = self.signMedia.media.get(cv2.CAP_PROP_FPS) if str(self.signMedia) == "cam" else 24
        pyglet.clock.schedule_interval(self.update, 1.0 / fps)
        super().run()

    def close(self):
        super().close()

    def __del__(self):
        pass
