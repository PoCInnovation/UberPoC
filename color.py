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
    reshape_area : Resize the rect to fit the size of the image
    
    Arguments : img_w, img_h : size of the image, Rect()
    Return : Rect
"""
def reshape_area(img_w, img_h, area):
    if area.x > img_w or area.y > img_h:
        raise ValueError
    if area.x + area.w > img_w:
        area.w = area.x + area.w - img_w
    if area.y + area.h > img_h:
        area.h = area.y + area.h - img_h
    print(area)
    return area


"""
    get_major_color : From a Pixel array, return the dominant color of a specific area
    
    Arguments : img_array : np.array, area: Rect
    Return : (R, G, B): tuple , covered_area: float
"""
def get_major_color(img_array:np.ndarray, area):
    max_w, max_h, nb_channels = img_array.shape
    dom_r, dom_g, dom_b = 0, 0, 0
    div = area.w * area.h

    for off_y in range(area.w):
        for off_x in range(area.h):
            pixel = img_array[area.y + off_y, area.x + off_x]
            dom_r += pixel[0]
            dom_g += pixel[1]
            dom_b += pixel[2]
    return dom_r / div, dom_g / div, dom_b / div

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
area = Rect(638, 478, 4, 4)
reshape_area(img_w, img_h, area)
print(get_major_color(obs, area))
img = Image.fromarray(obs)
img.save("test.png")

pyglet.app.run()
env.close()