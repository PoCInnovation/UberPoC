#! /usr/bin/env python3

import argparse
import sys

class App:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def config_parser(self):
        self.parser.add_argument("--video-name", default=None)
        self.parser.add_argument("--duckietown", default=False, action="store_true")
        sign_detection_subparser = self.parser.add_subparsers()
        parser_sign = sign_detection_subparser.add_parser('sign_detection')
        parser_sign.add_argument('--cam', action="store_true", default=None)
        parser_sign.add_argument("--img", default=None)
        return self

    def parse(self):
        args = self.parser.parse_args()
        if sys.argv[1] == "sign_detection":
            from visualizer.SignVisualizer import SignVisualizer
            print(args)
            if args.cam:
                vis = SignVisualizer(target="cam")
            if args.img is not None:
                vis = SignVisualizer(target=args.img)
        else:
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



