import tensorflow as tf
import pandas as pd
import numpy as np

from trainer.config import label_name

class Model:
    def create_model(self):
        raise NotImplementedError

class LogisticRegression(Model):
    def __init__(self):
        pass
    def create_model(self, training= None):
        inputs = tf.keras.Input(
                    shape= (4,), name= 'input')
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.Dense(
            int(32), activation='relu', name='fc1')(x)
        x = tf.keras.layers.Dense(
            int(32), activation='relu', name='fc2')(x)

        outputs = tf.keras.layers.Dense(3, activation='softmax', name=label_name)(x)
        model = tf.keras.Model(inputs= inputs, outputs= outputs)

        model.compile(
            optimizer= 'adam',
            loss= 'categorical_crossentropy',
            metrics=['accuracy'])

        return model

if __name__ == "__main__":
    pass





