import sys
import pyglet
from pyglet.window import key
from pyglet.gl import *
from PIL import Image

import numpy as np

import gym
import gym_duckietown
import cv2


class Visualizer(pyglet.window.Window):

    def __init__(self, duckietown = False):
        super().__init__(640, 480, visible=False)
        self.duckietown = duckietown
        self.frame_rate = 24
        self.result = None

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()
            sys.exit(0)

    def on_close(self):
        self.close()
        sys.exit(0)

    def cv2glet(self, img):
        rows, cols, channels = img.shape
        raw_img = Image.fromarray(img).tobytes()
        top_to_bottom_flag = -1
        bytes_per_row = channels * cols
        pyimg = pyglet.image.ImageData(width=cols,
                                       height=rows,
                                       format='BGR',
                                       data=raw_img,
                                       pitch=top_to_bottom_flag * bytes_per_row)
        return pyimg

    def on_draw(self):
        if self.result is not None:
            self.cv2glet(cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)).blit(0, 0)

    def show(self, result):
        self.result = result

    def run(self):
        self.set_visible(True)
        pyglet.app.run()


class ControlledVisualizer(Visualizer):

    def __init__(self, env_name):
        super().__init__(duckietown=True)
        self.obs = None
        self.env = gym.make(env_name)

        self.key_handler = key.KeyStateHandler()
        self.push_handlers(self.key_handler)
        self.reset()

    def reset(self):
        self.env.reset()

    def show(self, obs):
        super().show(obs)

    def update(self, dt):
        action = np.array([0, 0])

        if self.key_handler[key.UP]:
            action = np.array([0.44, 0.0])
        if self.key_handler[key.DOWN]:
            action = np.array([-0.44, 0])
        if self.key_handler[key.LEFT]:
            action = np.array([0.35, +1])
        if self.key_handler[key.RIGHT]:
            action = np.array([0.35, -1])
        if self.key_handler[key.SPACE]:
            action = np.array([0, 0])

        obs, reward, done, info = self.env.step(action)
        if done:
            self.reset()
        self.show(obs)

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1.0 / self.env.unwrapped.frame_rate)
        super().run()

    def __del__(self):
        self.pop_handlers()
        self.env.close()


class VideoVisualizer(Visualizer):

    def __init__(self, file):
        super().__init__(duckietown=False)
        self.cap = None

    def show(self, obs):
        result = obs
        super().show(result)

    def on_key_press(self, symbol, modifiers):
        super().on_key_press(symbol, modifiers)

    def update(self, dt):
        a = np.array([])
        self.show(a)

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1.0 / self.frame_rate)
        super().run()
