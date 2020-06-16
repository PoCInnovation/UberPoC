#! /usr/bin/env python3

import sys
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

import numpy as np
from numpy import asarray
import pyglet
from pyglet.window import key
from PIL import Image
import cv2

class ImageZone:
    def __init__(self, x, y, w, h, arr: np.ndarray):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.arr = arr[self.y:self.h, self.x:self.w]
        self.palette = list()

    """
        reshape : Resize the rect to fit the size of the image

        Arguments : img_w, img_h
        Return : self
    """

    def get_palette(self, nb_colors):
        paletted = Image.fromarray(self.arr).quantize(colors=nb_colors)
        self.palette = [[paletted.getpalette()[j * 3 + i] for i in range(3)] for j in range(nb_colors)]
        return self.palette

    def normalize(self):
        self.arr = cv2.cvtColor(self.arr, cv2.COLOR_RGB2HSV)
        cv2.imwrite("test.png", self.arr)

    def __str__(self):
        return f"ImageZone: x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}"


# env = gym.make("Duckietown-udem1-v0")
#
# env.reset()
# env.render()
#
#
# @env.unwrapped.window.event
# def on_key_press(symbol, modifier):
#     if symbol == key.ESCAPE:
#         env.close()
#         sys.exit(0)

# key_handler = key.KeyStateHandler()
# env.unwrapped.window.push_handlers(key_handler)
#
# action = np.array([0.3, 0.00])
# obs, reward, done, info = env.step(action)
# env.render()
# img_h, img_w, nb_channels = obs.shape
# print(obs.shape)
# area = ImageZone(0, 0, 640, 480, obs)
# area.normalize()
# pyglet.app.run()
# env.close()

cap = cv2.VideoCapture("Full Self-Driving.mp4")
if cap.isOpened() is False:
    sys.exit(1)
while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break
    area = ImageZone(0, 0, 720, 480, frame)
    area.normalize()
    cv2.imshow("Vid", area.arr)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()