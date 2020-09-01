#!/usr/bin/env python3

import os
from pyimagesearch import config
from imutils import paths
from bs4 import BeautifulSoup
import cv2

class DatasetBuilder:
    def __init__(self):
        pass

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

            gtBoxes.append((xMin, yMin, xMax, yMax)
        return gtBoxes

    def processImages(self):
        totals = [0] * len(config.LABELS_PATH)
        imagePaths = list(paths.list_images(config.ORIG_IMAGES))
        for (i, imagePath) in enumerate(imagePaths):
            print("[+] processing image {}/{}...".format(i + 1, len(imagePaths)))
            fname = imagePath.split(os.path.sep)[-1]
            fname = fname[:fname.rfind(".")]
            annotPath = os.path.sep.join([config.ORIG_ANNOTS, f"{fname}.xml"])
            gtBoxes = self.parseXMLAnnotations(annotPath)

            image = cv2.imread(imagePath)
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            rects = ss.process()

            proposedRect = []
            for (x, y, w, h) in rects:
                proposedRect.append((x, y, x + w, y + h))

    def prepare_dataset(self):
        for _dir in config.LABELS_PATH:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        print("[+] Dataset folders created")
        self.processImages()


datasetB = DatasetBuilder()
datasetB.prepare_dataset()