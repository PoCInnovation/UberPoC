#!/usr/bin/env python3
from app import App

try:
    app = App()
    app.config_parser().parse()
except ValueError as e:
    print(e)
except FileNotFoundError as e:
    print(e)