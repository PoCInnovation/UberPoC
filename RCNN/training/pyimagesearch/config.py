import os

ORIG_BASE_PATH = "signs"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])

BASE_PATH = "dataset"
LABELS = ["stop", "nothing"]
LABELS_PATH = [os.path.sep.join([BASE_PATH, label]) for label in LABELS]

MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

MAX_POSITIVE = 30
MAX_NEGATIVE = 10

INPUTS_DIMS = (224,224)

MODEL_PATH = "sing_detector.h5"
ENCODER_PATH = "label_encoder.pickle"
MIN_PROBA = 0.99