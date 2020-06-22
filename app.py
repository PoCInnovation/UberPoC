#! /usr/bin/env python3

import sys
import argparse
import numpy as np
import cv2
from visualizer import *

class App:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def config_parser(self):
        self.parser.add_argument("--video-name", default=None)
        self.parser.add_argument("--duckietown", default=False, action="store_true")
        return self

    def parse(self):
        args = self.parser.parse_args()
        if args.duckietown:
            duckietown_vis = ControlledVisualizer("Duckietown-udem-v0")
            duckietown_vis.run()
        elif args.video_name is not None:
            video_vid = VideoVisualizer(args.video_name)
            video_vid.run()
        else:
            raise ValueError("No valid arguments passed")

    @staticmethod
    def help():
        print("Image treatment Visualizer")
        print("Usage: ./app.py [--video_name path_to_vid | --duckietown]")

if __name__ == '__main__':
    try:
        app = App()
        app.config_parser().parse()
    except ValueError as e:
        print(e)
        App.help()



