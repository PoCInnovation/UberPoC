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


class ImageZone:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    """
        reshape : Resize the rect to fit the size of the image

        Arguments : img_w, img_h
        Return : self
    """
    def reshape(self, img_w, img_h):
        if self.x > img_w or self.y > img_h:
            raise ValueError
        if self.x + self.w > img_w:
            self.w = self.x + self.w - img_w
        if self.y + self.h > img_h:
            self.h = self.y + self.h - img_h
        print(self)
        return self

    """
        get_major_color : From a Pixel array, return the dominant color of a specific area

        Arguments : img_array : np.array, area: ImageZone
        Return : (R, G, B): tuple , covered_area: float
    """
    def get_major_color(self, img_array: np.ndarray):
        dom_r, dom_g, dom_b = 0, 0, 0
        div = self.w * self.h

        for off_y in range(self.w):
            for off_x in range(self.h):
                pixel = img_array[self.y + off_y, self.x + off_x]
                dom_r += pixel[0]
                dom_g += pixel[1]
                dom_b += pixel[2]
        return dom_r / div, dom_g / div, dom_b / div

    def __str__(self):
        return f"ImageZone: x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}"


env = gym.make("Duckietown-udem1-v0")

env.reset()
env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifier):
    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.3, 0.00])
obs, reward, done, info = env.step(action)
env.render()
img_h, img_w, nb_channels = obs.shape
area = ImageZone(638, 478, 4, 4)
area.reshape(img_w, img_h)
print(area.get_major_color(obs))
img = Image.fromarray(obs)
img.save("test.png")

pyglet.app.run()
env.close()