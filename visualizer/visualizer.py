import sys
import pyglet
from pyglet.window import key
from pyglet.gl import *
from PIL import Image
import cv2


class Visualizer(pyglet.window.Window):

    def __init__(self, duckietown = False):
        super().__init__(640, 480, visible=False)
        self.duckietown = duckietown
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
            print(self.masks["lines"])
        # N pout activer/désactiver  l'affichage de l'image normalisée
        if symbol == key.N:
            self.masks["normalized"] = self.masks["normalized"] is not True
            print(self.masks["normalized"])

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
        # Fait tout tes traitements d'images en vérifiant si les masks sont appliquées
        if self.masks["lines"] is True:
            pass
        if self.masks["normalized"] is True:
            pass
        # Il faut merge les tables

        # Puis tu stocke le résultat à afficher dans self.result et c'est bon
        self.result = result

    def run(self):
        self.set_visible(True)
        pyglet.app.run()
