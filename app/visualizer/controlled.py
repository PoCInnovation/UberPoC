from . import Visualizer

import pyglet
from pyglet.window import key
from pyglet.gl import *

import numpy as np
from gym_duckietown.envs import DuckietownEnv
import gym


class ControlledVisualizer(Visualizer):

    def __init__(self, env_name):
        super().__init__(duckietown=True)
        self.set_caption(f"Duckietown - Visualizer")
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
        action = np.array([0.0, 0.0])

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
