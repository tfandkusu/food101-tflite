import numpy as np
import tensorflow as tf
import data
from mobilenet import MobileNet

def make(tflite = False):
    "モデルを作成する"
    if tflite:
        # TensorFlow Lite用に改造したMobileNet
        return MobileNet(
            input_shape=(224,224,3),
            alpha=0.5,weights=None, classes=101)
    else:
        # TensorFlow標準のMobileNet
        return tf.keras.applications.MobileNet(
            input_shape=(224,224,3),
            alpha=0.5,weights=None, classes=101)
