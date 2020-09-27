import numpy as np


class FeatureExtractor:
    def __init__(self):
        pass

    def extract(self, *args, **kwargs):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
        return np.random.rand(4096)
