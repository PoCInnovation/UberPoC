import sys
from visualizer import visualizer

import pyglet
from pyglet.window import key
from pyglet.gl import *

import numpy as np
import cv2


class VideoVisualizer(visualizer.Visualizer):

    def __init__(self, file):
        super().__init__(duckietown=False)
        self.file = file
        self.cap = cv2.VideoCapture(file)
        if self.cap.isOpened() is False:
            raise FileNotFoundError(f"{file}: file not found")
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.set_size(self.w, self.h)
        self.set_caption(f"{file} - Visualizer")

    def show(self, obs):
        result = obs
        super().show(result)

    def update(self, dt):
        if self.cap.isOpened() is False:
            sys.exit(1)
        ret, frame = self.cap.read()
        if ret is False:
            self.close()
        self.show(frame)

    def run(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        pyglet.clock.schedule_interval(self.update, 1.0 / fps)
        super().run()

    def __del__(self):
        self.cap.release()
