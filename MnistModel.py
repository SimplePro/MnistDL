from keras.models import load_model
from PIL import Image
import numpy as np


def predict(filepath=None):
    if filepath is None:
        raise Exception("filepath most not be None")
    image = ((np.array(np.resize(Image.open(filepath).convert("L"), (1, 784))) / 255) - 1) * -1
    mnist_model = load_model("./mnist_dl.h5")
    result = mnist_model.predict_classes(image)[0]
    return result
