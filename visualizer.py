import pyglet
from pyglet.window import key

import gym
import numpy as np
import cv2

class Visualizer:

    def __init__(self, duckietown = False):
        self.duckietown = duckietown
        self.key_handler = key.KeyStateHandler()

    def run(self):
        pyglet.app.run()


class ControlledVisualizer(Visualizer):

    def __init__(self, env_name):
        super().__init__(duckietown=True)
        self.obs = None
        self.env = gym.make(env_name)
        self.env.unwrapped.window.push_handlers(self.key_handler)
        self.action = np.array([])

    def reset(self):
        self.env.reset()

    def show(self, obs):
        pass

    def update(self, dt):
        action = np.array([0.0, 0.0])
        obs, reward, done, info = self.env.step(action)
        self.show(obs)

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1.0 / self.env.unwrapped.frame_rate)
        super().run()

    def __del__(self):
        self.env.close()


class VideoVisualizer(Visualizer):

    def __init__(self, file):
        super().__init__(duckietown=False)
        self.cap = None
        #self.master = pyglet.window.Window()

    def show(self, obs):

    def update(self, dt):
        self.show(array)

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1.0 / self.master.frame_rate)
        super().run()
