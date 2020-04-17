import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from tensorflow import keras
from sklearn.utils import shuffle
from tqdm import tqdm

def CreateModel():
    # Create a sequantial model
    model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation = 'relu', padding='valid', bias_initializer='glorot_uniform', input_shape=(28, 28, 1)),
    # keras.layers.Conv2D(32, (3,3), activation = 'relu', padding='valid', bias_initializer='glorot_uniform'),
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation='sigmoid', bias_initializer='glorot_uniform'),
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

    features_expanded = tf.expand_dims(feature_vectors, 0)
    centroids_expanded = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(tf.square(tf.subtract(features_expanded, centroids_expanded)), 2)

    index = tf.math.argmin(distances, 0)

    tf.print("=======Done assigning y_true!=======")
    return tf.gather(centroids, index)

def RecalculateCentroids(centroids, feature_vectors, y_true):
    # Get all feature vectors assigned to centroid, by comparing the centroid from y_true
    # Replace that centroid with the mean of all the feature vectors assigned to that centroid
    new_centroids = []

    for count, centroid in enumerate(centroids):

        keep = tf.reduce_all(tf.math.equal(y_true, centroid), axis=1)
        tf.print(tf.reduce_sum(tf.cast(keep, tf.float32)), "Samples in centroid", count)


        x_temp = feature_vectors[keep]

        new_centroids.append(tf.reduce_mean(x_temp, axis = 0))

    tf.print("=======Done recalculating centroids!=======")
    return tf.stack(new_centroids)

def EvaluateModel(x_test, y_test, model, centroids):

    feature_vectors = CreateFeatureVectors(model, x_test)
    features_expanded = tf.expand_dims(feature_vectors, 0)
    centroids_expanded = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(tf.square(tf.subtract(features_expanded, centroids_expanded)), 2)

    predictions = tf.math.argmin(distances, 0)

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
        feature_vectors.append(model(x))

    return tf.concat(feature_vectors, axis = 0)

# %% Main
if __name__ == "__main__":

    num_epochs = 100
    batch_size = 32

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
    feature_vectors = CreateFeatureVectors(model, x_train, batch_size)

    # %% Get one feature vector per class to initialize centroids
    centroids = InitializeCentroids(feature_vectors, y_train)

    # %% Create a vector of targets for each image, y_true = the closest centroid to each image
    y_true = AssignTargets(feature_vectors, centroids)

    # %% Adjust centroids (?)
    # centroids = RecalculateCentroids(centroids, feature_vectors, y_true)

    # %% Initialize Training parameters
    train_loss_results = []
    train_accuracy_results = []

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    batches = len(x_train)/batch_size

    # %% Start training
    for epoch in range(num_epochs):

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_mse = tf.keras.metrics.MeanSquaredError()

        # Put x_train and y_true into a dataset object and divide into batches
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_true))
        dataset = dataset.batch(batch_size)

        i = 0
        # Training loop - using batches of 32
        with tqdm(total=batches) as pbar:
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

                pbar.update(1)

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_mse.result())

        # Recalculate feature vectors
        feature_vectors = CreateFeatureVectors(model, x_train, batch_size)

        # Reassign y_true for each x_train
        y_true = AssignTargets(feature_vectors, centroids)

        # # Recalculate Centroids
        # centroids = RecalculateCentroids(centroids, feature_vectors, y_true)

        # Print results per epoch
        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.6f}, MSE: {:.6f}".format(epoch,
                                                                   epoch_loss_avg.result(),

                                                                   epoch_mse.result()))
        # Evaluate Model during training. This is just a test to see what the behavior is during training
        EvaluateModel(x_test, y_test, model, centroids)


    # print("Done training! Saving Weights")
    # model.save("saved_model")



    print("=============================Done!=============================")






