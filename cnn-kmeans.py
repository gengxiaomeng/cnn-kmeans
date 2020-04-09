import os
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.utils import shuffle

def CreateModel():
    # Create a sequantial model
    model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation = 'relu', padding='valid', bias_initializer='glorot_uniform', input_shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3,3), activation = 'relu', padding='valid', bias_initializer='glorot_uniform'),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='tanh', bias_initializer='glorot_uniform'),
    ])

    return model

def InitializeCentroids(feature_vectors, y_train):
    # Get one centroid per class
    i = 0
    centroids = []
    
    for index, label in enumerate(y_train):
        if label == i:
            centroids.append(feature_vectors[index])
            i += 1

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

def AssignTargets(feature_vectors, centroids):
    # Assign y_true for each vector where y_true = the closest centroid
    y_true = []
        
    for feature_vector in feature_vectors:
            
        l2_norm = tf.norm(feature_vector - centroids, axis = 1)
        index = tf.math.argmin(l2_norm)
        y_true.append(centroids[index])
    
    tf.print("=======Done assigning y_true!=======")
    return tf.stack(y_true)

def RecalculateCentroids(centroids, feature_vectors, y_true):
    # Get all feature vectors assigned to centroid, by comparing the centroid from y_true
    # Replace that centroid with the mean of all the feature vectors assigned to that centroid
    new_centroids = []
    
    for count, centroid in enumerate(centroids):
        
        keep = tf.reduce_any(tf.math.equal(y_true, centroid), axis=1)
        tf.print(tf.reduce_sum(tf.cast(keep, tf.float32)), "Samples in centroid", count)
        

        x_temp = feature_vectors[keep]
        
        new_centroids.append(tf.reduce_mean(x_temp, axis = 0))
        
    tf.print("=======Done recalculating centroids!=======")
    return tf.stack(new_centroids)
           
    
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
    
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    # %% Extract feature vectors from the training set
    feature_vectors  = model(x_train)
    
    # %% Get one feature vector per class to initialize centroids
    centroids = InitializeCentroids(feature_vectors, y_train)
        
    # %% Create a vector of targets for each image, y_true = the closest centroid to each image
    y_true = AssignTargets(feature_vectors, centroids)
    
    # %% Adjust centroids (?)
    # centroids = RecalculateCentroids(centroids, feature_vectors, y_true)
    RecalculateCentroids(centroids, feature_vectors, y_true)
    # %% Initialize Training parameters
    train_loss_results = []
    train_accuracy_results = []
    
    num_epochs = 100
    batch_size = 16
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    
    # %% Start training
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_mse = tf.keras.metrics.MeanSquaredError()
        
        # Put x_train and y_true into a dataset object and divide into batches
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
            epoch_mse.update_state(y, model(x, training=True))
    
        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_mse.result())
          
        # Recalculate feature vectors
        feature_vectors = model(x_train)
        
        # Recalculate Centroids
        centroids = RecalculateCentroids(centroids, feature_vectors, y_true)

        # Reassign y_true for each x_train
        y_true = AssignTargets(feature_vectors, centroids)
        
        # Print results per epoch
        if epoch % 1 == 0:
          print("Epoch {:03d}: Loss: {:.6f}, MSE: {:.6f}".format(epoch,
                                                                      epoch_loss_avg.result(),
                                                                      epoch_mse.result()))
    
    



    
