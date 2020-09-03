#!/usr/bin/env python3

import os
from pyimagesearch import config
from pyimagesearch.iou import compute_iou
from imutils import paths
from bs4 import BeautifulSoup
import cv2

class DatasetBuilder:
    def __init__(self):
        self.counters = [{"label": label, "counter": 0} for label in config.LABELS]
        print (self.counters)

    def get_counter_index(self, label):
        for i in range(len(self.counters)):
            if self.counters[i]["label"] == label:
                return i
        return -1

    def parseXMLAnnotations(self, annothPath):
        contents = open(annothPath).read()
        soup = BeautifulSoup(contents, "html.parser")
        gtBoxes = []

        w = int(soup.find("width").string)
        h = int(soup.find("height").string)

        for o in soup.find_all("object"):
            label = o.find("name").string
            xMin = max(0, int(o.find("xmin").string))
            yMin = max(0, int(o.find("ymin").string))
            xMax = min(w, int(o.find("xmax").string))
            yMax = min(h, int(o.find("ymax").string))

            gtBoxes.append((label, xMin, yMin, xMax, yMax))
        return gtBoxes

    def runSS(self, image):
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()

        proposedRect = []
        for (x, y, w, h) in rects:
            proposedRect.append((x, y, x + w, y + h))
        return proposedRect

    def isFullOverlap(self, gtBox, proposedRect):
        (propStartX, propStartY, propEndX, propEndY) = proposedRect
        (label, gtStartX, gtStartY, gtEndX, gtEndY) = gtBox

        fullOverlap = propStartX >= gtStartX
        fullOverlap = fullOverlap and propStartY >= gtStartY
        fullOverlap = fullOverlap and propEndX <= gtEndX
        fullOverlap = fullOverlap and propEndY <= gtEndY
        return fullOverlap

    def processImages(self):
        imagePaths = list(paths.list_images(config.ORIG_IMAGES))
        for (i, imagePath) in enumerate(imagePaths):
            print("[+] processing image {}/{}...".format(i + 1, len(imagePaths)))
            fname = imagePath.split(os.path.sep)[-1]
            fname = fname[:fname.rfind(".")]
            annotPath = os.path.sep.join([config.ORIG_ANNOTS, f"{fname}.xml"])
            gtBoxes = self.parseXMLAnnotations(annotPath)

            image = cv2.imread(imagePath)
            proposedRects = self.runSS(image)
            positiveROIs = 0
            negativeROIs = 0

            if len(gtBoxes) == 0:
                filename = f"{self.counters[self.get_counter_index('nothing')]['counter']}.png"
                outputPath = os.path.sep.join([config.LABELS_PATH[self.get_counter_index('nothing')], filename])
                roi = cv2.resize(image, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, roi)
                self.counters[self.get_counter_index('nothing')]['counter'] += 1
            else:
                for proposedRect in proposedRects:
                    (propStartX, propStartY, propEndX, propEndY) = proposedRect

                    for gtBox in gtBoxes:
                        iou = compute_iou(gtBox[1:], proposedRect)
                        (label, gtStartX, gtStartY, gtEndX, gtEndY) = gtBox
                        filename = f"{self.counters[self.get_counter_index('nothing')]['counter']}.png"
                        negativeROIs += 1

                        roi = None
                        outputPath = None

                        if iou > 0.7 and positiveROIs <= config.MAX_POSITIVE:
                            roi = image[propStartY:propEndY, propStartX:propEndX]
                            filename = f"{self.counters[self.get_counter_index(label)]['counter']}.png"
                            outputPath = os.path.sep.join([config.LABELS_PATH[self.get_counter_index(label)], filename])
                            positiveROIs += 1
                            self.counters[self.get_counter_index(label)]['counter'] += 1

                        if not self.isFullOverlap(gtBox, proposedRect) and iou < 0.05 and negativeROIs <= config.MAX_NEGATIVE:
                            roi = image[propStartY:propEndY, propStartX:propEndX]
                            filename = f"{self.counters[self.get_counter_index('nothing')]['counter']}.png"
                            outputPath = os.path.sep.join([config.LABELS_PATH[self.get_counter_index(label)], filename])
                            negativeROIs += 1

                        if roi is not None and outputPath is not None:
                            roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                            cv2.imwrite(outputPath, roi)

    def prepare_dataset(self):
        for _dir in config.LABELS_PATH:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        print("[+] Dataset folders created")
        self.processImages()


datasetB = DatasetBuilder()
datasetB.prepare_dataset()