#!/usr/bin/env python3

from . import Visualizer

import pyglet
from pyglet.window import key
from pyglet.gl import *

import cv2
import imutils

class HumanVisualiser(Visualizer):
    
    def __init__(self, target):
        super(HumanVisualiser, self).__init__(duckietown=False)
        try:
            self.HumanMedia = HumanMedia(target)
        except FileNotFoundError:
            print("Couldn't open the HumanMedia")
            sys.exit(1)

    def update(self, dt):
        frame = self.HumanMedia.getImageWithHumanDetection()
        self.show(frame)

    def show(self, obs):
        result = obs
        super().show(result)

    def on_key_press(self, symbol, identifiers):
        return

    def run(self):
        fps = 0.9
        pyglet.clock.schedule_interval(self.update, 1.0 / fps)
        super().run()

    def close(self):
        super().close()

    def __del__(self):
        pass


class HumanMedia:
    def __init__(self, target):
        self.media = cv2.imread(target)
        if self.media is None:
            raise FileNotFoundError
        self.target = target

    def getImageWithHumanDetection(self):
        width = 640
        height = 480
        frame = self.media.copy()
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        frame = imutils.resize(frame, width=min(1500, frame.shape[1]))
        (regions, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.10)
        for (x, y, w, h) in regions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        dsize = (width, height)
        frame = cv2.resize(frame, dsize)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def __str__(self):
        return self.target



    def isOpened(self):
        return self.media.isOpened()