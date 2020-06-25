#! /usr/bin/env python3

import argparse

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
            from visualizer.controlled import ControlledVisualizer
            vis = ControlledVisualizer("Duckietown-udem1-v0")
        elif args.video_name is not None:
            from visualizer.video import VideoVisualizer
            vis = VideoVisualizer(args.video_name)
        else:
            self.help()
            raise ValueError("No valid arguments passed")
        vis.run()

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
    except FileNotFoundError as e:
        print(e)



