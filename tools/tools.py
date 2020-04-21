# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:19:42 2020

@author: Juicebox
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from tensorflow import keras

def CreateModel():
    # Create a sequantial model
    model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation = 'relu', padding='valid', bias_initializer='glorot_uniform', input_shape=(28, 28, 1)),
    # keras.layers.Conv2D(32, (3,3), activation = 'relu', padding='valid', bias_initializer='glorot_uniform'),
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation='sigmoid', bias_initializer='glorot_uniform'),
    ])

    return model

def InitializeCentroids(feature_vectors, y_train, num_classes = 10):
    # Get one centroid per class
    i = 0
    centroids = []

    for i in range(10):
        index = y_train.tolist().index(i)
        centroids.append(feature_vectors[index])

    return tf.stack(centroids)

def CalculateLoss(model, x, y, training):
    # Calculate MSE loss between y_true and prediction.
    loss_object = tf.keras.losses.MeanSquaredError()

    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)

def CalculateGradients(model, inputs, targets):
    # Caclulate the gradients with respect to the loss

    with tf.GradientTape() as tape:
        loss_value = CalculateLoss(model, inputs, targets, training=True)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def AssignTargets(feature_vectors, centroids, batch_size = 32):
    # Assign y_true for each vector where y_true = the closest centroid
    index = []

    feature_dataset = tf.data.Dataset.from_tensor_slices(feature_vectors)
    feature_dataset = feature_dataset.batch(batch_size)

    for feature_vector in feature_dataset:
        features_expanded = tf.expand_dims(feature_vector, 0)
        centroids_expanded = tf.expand_dims(centroids, 1)

        distances = tf.reduce_sum(tf.square(tf.subtract(features_expanded, centroids_expanded)), 2)

        index.extend(tf.math.argmin(distances, 0))

    return tf.gather(centroids, index), index

def RecalculateCentroids(centroids, feature_vectors, y_true):
    # Get all feature vectors assigned to centroid, by comparing the centroid from y_true
    # Replace that centroid with the mean of all the feature vectors assigned to that centroid
    new_centroids = []

    for count, centroid in enumerate(centroids):

        keep = tf.reduce_all(tf.math.equal(y_true, centroid), axis=1)
        tf.print(tf.reduce_sum(tf.cast(keep, tf.float32)), "Samples in centroid", count)

        x_temp = feature_vectors[keep]

        new_centroids.append(tf.reduce_mean(x_temp, axis = 0))

    print("================== Done recalculating centroids! ==================")
    return tf.stack(new_centroids)

def EvaluateModel(x_test, y_test, model, centroids, batch_size = 32):

    feature_vectors = CreateFeatureVectors(model, x_test)

    _, predictions = AssignTargets(feature_vectors, centroids, batch_size)

    correct_predictions = tf.reduce_sum(tf.cast(tf.math.equal(y_test, predictions), tf.float32))

    total_accuracy = correct_predictions/y_test.shape[0]

    print("Pseudo-accuracy: {}".format(total_accuracy))

    # Create confusion matrix
    confusion_matrix = tf.math.confusion_matrix(y_test, predictions, num_classes = 10).numpy()

    # Plot Confusion Matrix
    df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
    df_cm.fillna(value=np.nan, inplace=True)
    plt.figure(figsize=(10,10))
    sn.set(font_scale=0.8) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 9}, fmt="d") # font size

    plt.show()

def CreateFeatureVectors(model, x_train, batch_size = 32):
    # Split the dataset into batches to make it trainable on the GPU
    x_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    x_dataset = x_dataset.batch(batch_size)

    feature_vectors = []

    for x in x_dataset:
        feature_vectors.append(model(x, training = False))

    return tf.concat(feature_vectors, axis = 0)

def AddFullyConnectedLayer(base_model):

    # x = keras.layers.Flatten()(base_model.output)
    x = keras.layers.Dense(98, activation='sigmoid', bias_initializer='glorot_uniform')(base_model.output)
    x = keras.layers.Dense(49, activation='sigmoid', bias_initializer='glorot_uniform')(x)
    x = keras.layers.Dense(25, activation='sigmoid', bias_initializer='glorot_uniform')(x)

    model = keras.Model(base_model.input, x)

    return model
