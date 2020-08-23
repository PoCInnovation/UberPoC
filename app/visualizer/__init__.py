import sys
import pyglet
from pyglet.window import key
from pyglet.gl import *
from PIL import Image
import cv2
import numpy as np

def Cropped(result):

    height, width = result.shape
    mask = np.zeros_like(result)
    polygon = np.array([[(0, height * 2 / 3), (width, height * 2 / 3), (width, height), (0, height), ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(result, mask)

    return cropped_edges

def Canny_cropped(frame, min_color):

    blur = cv2.bilateralFilter(frame,9,75,75)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 0, min_color])
    upper_red = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    edges = cv2.Canny(mask, 200, 400)
    cropped_edges = Cropped(edges)

    return cropped_edges

def HoughLine(cropped_edges, threshold):

    rho = 1
    angle = np.pi / 180
    min_threshold = threshold
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=1, maxLineGap=2000)

    return line_segments

def edge_line(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 2 / 3)
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))

    return [[x1, y1, x2, y2]]

def average_lines(frame, line_segments):

    if line_segments is None:
        return []
    height, width, _ = frame.shape
    left_line, right_line, lane_lines = [], [], []
    left_part = width * (1 - (1 / 3))
    right_part = width * (1 / 3)

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            angle = fit[0]
            intercept = fit[1]
            if angle < 0:
                if x1 < left_part and x2 < left_part:
                    left_line.append((angle, intercept))
            else:
                if x1 > right_part and x2 > right_part:
                    right_line.append((angle, intercept))
    if len(left_line) > 0:
        left_average = np.average(left_line, axis=0)
        lane_lines.append(edge_line(frame, left_average))
    if len(right_line) > 0:
        right_average = np.average(right_line, axis=0)
        lane_lines.append(edge_line(frame, right_average))

    return lane_lines

def display_lines(frame, lines):

    width = 2
    color = (0, 255, 0)
    img = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, width)
    img = cv2.addWeighted(frame, 0.8, img, 1, 1)

    return img

class Visualizer(pyglet.window.Window):

    def __init__(self, duckietown=False, sign=False):
        super().__init__(640, 480, visible=False)
        self.duckietown = duckietown
        self.sign = sign
        self.frame_rate = 24
        self.result = None
        self.masks = {"lines": False, "normalized": False}

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()
            sys.exit(0)
        # L pour activer/désactiver l'affichage des lignes
        if symbol == key.L:
            self.masks["lines"] = self.masks["lines"] is not True
        # N pout activer/désactiver  l'affichage de l'image normalisée
        if symbol == key.N:
            self.masks["normalized"] = self.masks["normalized"] is not True

    def on_close(self):
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
        if self.sign is False:
            if self.duckietown is True:
                frame = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                if self.masks["lines"] is True:
                    cropped = Canny_cropped(frame, 145)
                    line_seg = HoughLine(cropped, 60)
                    average = average_lines(frame, line_seg)
                    result = display_lines(result, average)
                if self.masks["normalized"] is True:
                    result = Canny_cropped(frame, 145)
            else:
                if self.masks["lines"] is True:
                    cropped = Canny_cropped(result, 225)
                    line_seg = HoughLine(cropped, 90)
                    average = average_lines(result, line_seg)
                    result = display_lines(result, average)
                if self.masks["normalized"] is True:
                    result = Canny_cropped(result, 225)
        else:
            pass
        self.result = result

    def run(self):
        self.set_visible(True)
        pyglet.app.run()
