import os
import autoencoder
import yaml
import tensorflow as tf

from tensorflow import keras
from sklearn.utils import shuffle
from tqdm import tqdm
from tools import tools

# %% Main
if __name__ == "__main__":
    
    tf.keras.backend.clear_session()
    
    # %% Read config file
    config_file = "./config/config.yaml"
    
    if not os.path.exists(config_file):
        print("Config file not found!")
        exit(1)
    else:
        with open(config_file, 'r') as f:
            args = yaml.safe_load(f)
            
    channels = args['channels']   
    classes = args['classes']
    trials = args['trials']     
    epochs = args['epochs']
    batch_size = args['batch_size']
    train_autoencoder = args['train_autoencoder']
    
    # %%Initialize Dataset
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    x_train = x_train/255.0
    x_test = x_test/255.0
   
    # %% Grab Model trained from autoencoder
    if not os.path.isdir('encoder_model') or train_autoencoder == True:
        print("Training autoencoder...")
        model = autoencoder.TrainAutoencoder(x_train, x_test, y_test, batch_size)
    else:
        print("Loading model...")
        model = tf.keras.models.load_model('encoder_model')
    
    model.summary()

    # %% Reshape dataset to fit model (samples, height, width, channels)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # %% Extract feature vectors from the training set
    feature_vectors = tools.CreateFeatureVectors(model, x_train, batch_size)

    # %% Get one feature vector per class to initialize centroids
    centroids = tools.InitializeCentroids(feature_vectors, y_train)

    # %% Create a vector of targets for each image, y_true = the closest centroid to each image
    y_true = tools.AssignTargets(feature_vectors, centroids)

    # %% Adjust centroids (?)
    centroids = tools.RecalculateCentroids(centroids, feature_vectors, y_true)

    # %% Initialize Training parameters
    train_loss_results = []
    train_accuracy_results = []

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    batches = len(x_train)/batch_size

    # %% Start training
    for epoch in range(epochs):

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
                loss_value, grads = tools.CalculateGradients(model, x, y)
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
        feature_vectors = tools.CreateFeatureVectors(model, x_train, batch_size)

        # Reassign y_true for each x_train
        y_true = tools.AssignTargets(feature_vectors, centroids)

        # # Recalculate Centroids
        # centroids = tools.RecalculateCentroids(centroids, feature_vectors, y_true)

        # Print results per epoch
        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.6f}, MSE: {:.6f}".format(epoch,
                                                                   epoch_loss_avg.result(),

                                                                   epoch_mse.result()))
        # Evaluate Model during training. This is just a test to see what the behavior is during training
        tools.EvaluateModel(x_test, y_test, model, centroids)


    # print("Done training! Saving Weights")
    # model.save("trained_kmeans_model")



    print("=============================Done!=============================")






