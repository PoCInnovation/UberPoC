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


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def __str__(self):
        return f"Rect: x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}"

env = gym.make("Duckietown-udem1-v0")

env.reset()
env.render()


"""
    get_major_color : From a Pixel array, 
    return the dominant color of a specific area
    Arguments : img_array : np.array, area: Rect
    Return : (R, G, B): tuple , covered_area: float
"""
def get_major_color(img_array:np.array, area):
    max_w, max_h, nb_channels = img_array.shape
    dom_r = 0
    dom_g = 0
    dom_b = 0
    divider = area.w * area.h
    for off_y in range(area.w):
        for off_x in range(area.h):
            print(img_array[area.x + off_x, area.y + off_y])

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
print(type(obs))
print(obs.shape)
get_major_color(obs, Rect(0, 0, 2, 2))
img = Image.fromarray(obs)
img.save("test.png")

pyglet.app.run()
env.close()