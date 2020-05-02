"""
Created on Fri Apr 17 15:19:42 2020

@author: Juicebox
"""
import os
import csv
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

def EvaluateModel(x_test, y_test, model, centroids, epoch, trial, batch_size=32,
                  save_matrix=False, training_set=True, save_directory=os.getcwd()):

    feature_vectors = CreateFeatureVectors(model, x_test)

    _, predictions = AssignTargets(feature_vectors, centroids, batch_size)

    correct_predictions = tf.reduce_sum(tf.cast(tf.math.equal(y_test, predictions), tf.float32))

    pseudo_accuracy = correct_predictions/y_test.shape[0]

    print("Pseudo-accuracy: {}".format(pseudo_accuracy))

    if save_matrix:

        save_directory = os.path.join(save_directory, "{}".format(trial))

        if os.path.isdir(save_directory) == False:
            os.mkdir(save_directory)

        # Create confusion matrix
        confusion_matrix = tf.math.confusion_matrix(y_test, predictions, num_classes = 10).numpy()

        # Plot Confusion Matrix
        df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
        df_cm.fillna(value=np.nan, inplace=True)
        plt.figure(figsize=(10,10))
        sn.set(font_scale=0.8) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 9}, fmt="d") # font size

        # plt.xlabel("Centroid")
        # plt.ylabel("y_true")

        if training_set:
            plt.savefig(os.path.join(save_directory, "Training Set Confusion Matrix Trial {} Epoch {}".format(trial, epoch)),
                        bbox_inches = 'tight', pad_inches = 0.1)
        else:
            plt.savefig(os.path.join(save_directory, "Test Set Confusion Matrix Trial {} Epoch {}".format(trial, epoch)),
                        bbox_inches = 'tight', pad_inches = 0.1)

        plt.close()
    return pseudo_accuracy

def ComputeCentroidDistances(centroids, trial, epoch, save_directory=os.getcwd()):
    # Compute the pairwise distance between centroid locations
    expanded_a = tf.expand_dims(centroids, 1)
    expanded_b = tf.expand_dims(centroids, 0)

    # This produces a matrix of pair-wise distances between the centroids
    distances = tf.reduce_sum(tf.math.squared_difference(expanded_a, expanded_b), 2).numpy()

    # Calculate average distance
    sum_distances = tf.reduce_sum(distances)/2
    average_distance = sum_distances/len(centroids)

    return average_distance

def CreateFeatureVectors(model, x_train, batch_size = 32):
    # Split the dataset into batches to make it trainable on the GPU
    x_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    x_dataset = x_dataset.batch(batch_size)

    feature_vectors = []

    for x in x_dataset:
        feature_vectors.append(model(x, training=False))

    return tf.concat(feature_vectors, axis = 0)

def AddFullyConnectedLayer(base_model):

    # x = keras.layers.Flatten()(base_model.output)
    x = keras.layers.Dense(98, activation='sigmoid', bias_initializer='glorot_uniform')(base_model.output)
    x = keras.layers.Dense(49, activation='sigmoid', bias_initializer='glorot_uniform')(x)
    x = keras.layers.Dense(25, activation='sigmoid', bias_initializer='glorot_uniform')(x)

    model = keras.Model(base_model.input, x)

    return model

def PlotTrainingResults(accuracy, loss, save_directory, trial):

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].grid(False)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(loss)

    axes[1].grid(False)
    axes[1].set_ylabel("Pseudo-accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(accuracy)

    save_directory = os.path.join(save_directory, "{}".format(trial))

    if os.path.isdir(save_directory) == False:
        os.mkdir(save_directory)

    plt.savefig(os.path.join(save_directory, "Training Results Trial {}".format(trial)),
                bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()

def PlotTestResults(accuracy, save_directory, trial):

    save_directory = os.path.join(save_directory, "{}".format(trial))

    if os.path.isdir(save_directory) == False:
        os.mkdir(save_directory)

    plt.figure()
    plt.grid(b=None)
    plt.plot(accuracy)
    plt.title('Accuracy on Test Set during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Pseudo-accuracy')
    plt.savefig(os.path.join(save_directory, "Test Results Trial {}".format(trial)),
                bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()

def PlotAverageCentroidDistance(centroid_distances, save_directory, trial):

    save_directory = os.path.join(save_directory, "{}".format(trial))

    if os.path.isdir(save_directory) == False:
        os.mkdir(save_directory)

    plt.figure()
    plt.grid(b=None)
    plt.plot(centroid_distances)
    plt.title('Average pair-wise centroid distance')
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.savefig(os.path.join(save_directory, "Average Centroid Distance Trial {}".format(trial)),
                bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()

def SaveValues(training_accuracy, test_accuracy, trial, results_save_file):

    with open(results_save_file, "a", newline="") as csv_file:
        scoreWriter = csv.writer(csv_file)
        scoreWriter.writerow([trial, training_accuracy, test_accuracy])

def PlotHistory(history, save_directory, trial):

    save_directory = os.path.join(save_directory, "{}".format(trial))

    if os.path.isdir(save_directory) == False:
        os.mkdir(save_directory)

    plt.figure()
    plt.grid(b=None)
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig(os.path.join(save_directory, "Autoencoder Loss History {}".format(trial)),
                bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()


