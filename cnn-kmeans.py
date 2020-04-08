import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.utils import shuffle

def CreateModel():

    model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation = 'relu', padding='valid', bias_initializer='glorot_uniform', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3,3), activation = 'relu', padding='valid', bias_initializer='glorot_uniform'),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
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

def InitializeCentroids(model, initial_images):

    centroids = model(initial_images)

    return centroids

def CalculateLoss(model, x, y, training):
    
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).

    loss_object = tf.keras.losses.MeanSquaredError()
    
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)

def CalculateGradients(model, inputs, targets):
    
    with tf.GradientTape() as tape:
        loss_value = CalculateLoss(model, inputs, targets, training=True)
        
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def AssignTargets(model, centroids, x_train):
    
    y_true = []
    
    feature_vectors = model(x_train)
    
    for feature_vector in feature_vectors:
            
        l2_norm = tf.norm(feature_vector - centroids, axis = 1)
        index = tf.math.argmin(l2_norm)
        y_true.append(centroids[index])
    
    return tf.stack(y_true)
    
# %% Main
if __name__ == "__main__":
    
    # Initialize Dataset
    
    tf.keras.backend.clear_session()
    
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    x_train = x_train/255.0
    x_test = x_test/255.0
    # %% Create Model with randomly initialized weights and biases
    model = CreateModel()
    
    # %% Reshape dataset to fit model (samples, height, width, channels)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # %% Get one image per class to initialize centroids
    initial_images = GetOneImagePerclass(x_train, y_train)
    
    # %% Initialize Centroids by passing it through the randomly initialized model
    centroids = InitializeCentroids(model, initial_images)
    
    # %% Create a vector of targets for each image, y_true = the closest centroid to them
    y_true = AssignTargets(model, centroids, x_train)
    
    # %% Initialize Training parameters
    train_loss_results = []
    train_accuracy_results = []
    
    num_epochs = 10
    samples_per_batch = 32
    batch_size =  tf.cast(x_train.shape[0]/samples_per_batch, tf.int64)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    
    # %% Start training
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.MeanSquaredError()
        
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_true))
        dataset = dataset.batch(batch_size)
        
        # Training loop - using batches of 32
        for x, y in dataset:

            # Optimize the model
            loss_value, grads = CalculateGradients(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model(x, training=True))
    
        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
          
        if epoch % 50 == 0:
          print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                      epoch_loss_avg.result(),
                                                                      epoch_accuracy.result()))
    
    



    
