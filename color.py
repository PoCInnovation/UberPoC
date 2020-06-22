#! /usr/bin/env python3

import sys
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import numpy as np
import pyglet
from pyglet.window import key
from PIL import Image
import cv2


class ImageZone:
    def __init__(self, x, y, w, h, arr: np.ndarray):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.full = arr
        self.arr = None
        self.reshape(x, y, w, h)
        self.img = Image.fromarray(self.arr)
        self.palette = list()

    def reshape(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.arr = self.full[self.y:self.h, self.x:self.w]

    def get_palette(self, nb_colors):
        paletted = Image.fromarray(self.arr).quantize(colors=nb_colors)
        self.palette = [[paletted.getpalette()[j * 3 + i] for i in range(3)] for j in range(nb_colors)]
        return self.palette, paletted

    def normalize(self):
        a = np.double(self.arr)
        b = a + 10
        c = np.uint8(b)
        self.arr = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
        cv2.imwrite("test.png", self.arr)

    def __str__(self):
        return f"ImageZone: x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}"


# cap = cv2.VideoCapture("Full Self-Driving.mp4")
# if cap.isOpened() is False:
#     sys.exit(1)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret is False:
#         break
#     area = ImageZone(0, 0, 720, 480, frame)
#     area.normalize()
#     cv2.imshow("Vid", area.arr)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
env = gym.make("Duckietown-udem1-v0")
env.reset()
#env.render()

action = np.array([0.3, 0.00])
obs, reward, done, info = env.step(action)
#env.render()
#pyglet.app.run()
print(obs.shape)
area = ImageZone(0,0,640,480, obs)
area.normalize()
while True:
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    cv2.imshow("tets", area.arr)
env.close()
