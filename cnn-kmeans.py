import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.utils import shuffle
def CreateModel():

    model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation = 'relu', padding='valid', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3,3), activation = 'relu', padding='valid'),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='tanh'),
    ])

    return model

def GetOneImagePerclass(x_train, y_train):

    i = 0
    initial_images = []

    for index, label in enumerate(y_train):
        if label == i:
            initial_images.append(x_train[index])
            i += 1

    return tf.stack(initial_images)

# def InitializeCentroids():


# def AssignTargets():


# def CalculateLoss():
    

if __name__ == "__main__":

    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    x_train = x_train/255.0
    x_test = x_test/255.0

    base_model = CreateModel()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    initial_images = GetOneImagePerclass(x_train, y_train);

    
    
