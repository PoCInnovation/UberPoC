from . import Visualizer

import pyglet
from pyglet.window import key
from pyglet.gl import *

import numpy as np
import cv2
import sys

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
        ### You need to apply the SignDetection AI on this part and return frame with the predict

        return frame

    def __str__(self):
        return self.target



    def isOpened(self):
        return self.media.isOpened()

class SignVisualizer(Visualizer):

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
