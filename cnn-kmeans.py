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
    learning_rate = args['learning_rate']
    use_autoencoder = args['use_autoencoder']
    train_autoencoder = args['train_autoencoder']
    autoencoder_learning_rate = args['autoencoder_learning_rate']
    autoencoder_epochs = args['autoencoder_epochs']
    save_encoder_model = args['save_encoder_model']
    save_folder_name = args['save_folder_name']

    save_directory = os.path.join(os.getcwd(), save_folder_name)

    results_save_file = os.path.join(save_directory, "Experiment Results.csv")

    if os.path.isdir(save_directory) == False:
        print("Creating save directory")
        os.makedirs(save_directory)

    for trial in range(trials):

        tf.keras.backend.clear_session()

        # %% Initialize Training parameters
        average_centroid_distance = []
        train_loss_results = []
        train_accuracy_results = []
        train_pseudo_accuracy = []
        test_pseudo_accuracy = []

        # %%Initialize Dataset
        mnist = keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)

        x_train = x_train/255.0
        x_test = x_test/255.0

        optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate)
        batches = len(x_train)/batch_size

        # %% Grab Model trained from autoencoder
        if use_autoencoder:
            if not os.path.isdir('encoder_model') or train_autoencoder == True:
                print("Training autoencoder...")
                model, history = autoencoder.TrainAutoencoder(x_train, x_test, y_test, batch_size, trial, autoencoder_epochs,
                                                     autoencoder_learning_rate, save_encoder_model, save_directory)

                tools.PlotHistory(history, save_directory, trial)

            else:
                print("Loading model...")
                model = tf.keras.models.load_model('encoder_model')

            # Disable training for convolutional layers and add dense layers
            for layer in model.layers:
                if 'conv2d' in layer.name:
                    layer.trainable = False

        # Use randomly initialized ConvNet instead
        else:
            model = tools.CreateModel()

        model.summary()

        # %% Reshape dataset to fit model (samples, height, width, channels)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

        # %% Extract feature vectors from the training set
        feature_vectors = tools.CreateFeatureVectors(model, x_train, batch_size)

        # %% Get one feature vector per class to initialize centroids
        centroids = tools.InitializeCentroids(feature_vectors, y_train)

        # %% Create a vector of targets for each image, y_true = the closest centroid to each image
        y_true, _ = tools.AssignTargets(feature_vectors, centroids, batch_size)

        # %% Adjust centroids (?)
        centroids = tools.RecalculateCentroids(centroids, feature_vectors, y_true)

        # %% Get centroid pairwise distance
        average_centroid_distance.append(tools.ComputeCentroidDistances(centroids, trial, 0, save_directory))

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
            y_true, _ = tools.AssignTargets(feature_vectors, centroids, batch_size)

            # Recalculate Centroids
            centroids = tools.RecalculateCentroids(centroids, feature_vectors, y_true)

            # Print results per epoch
            if epoch % 1 == 0:
                print("Epoch {:03d}: Loss: {:.6f}, MSE: {:.6f}".format(epoch,
                                                                       epoch_loss_avg.result(),
                                                                       epoch_mse.result()))

                # Evaluate Model during training. This is just a test to see what the behavior is during training
                train_pseudo_accuracy.append(tools.EvaluateModel(x_train, y_train, model, centroids, epoch,
                                                                  trial, batch_size, False, True, save_directory))

                # Evaluate Model on test set during training. This is just a test to see what the behavior is during training
                test_pseudo_accuracy.append(tools.EvaluateModel(x_test, y_test, model, centroids, epoch,
                                                                trial, batch_size, False, False, save_directory))

                # Evaluate distance between centroids
                average_centroid_distance.append(tools.ComputeCentroidDistances(centroids, trial, 0, save_directory))


        print("Plotting and Saving training results")
        # tools.PlotTrainingResults(train_pseudo_accuracy, train_loss_results, save_directory, trial)
        tools.EvaluateModel(x_train, y_train, model, centroids, epoch,
                                                         trial, batch_size, True, True, save_directory)

        print("Done training! Plotting accuracy on test Set")
        tools.PlotTestResults(test_pseudo_accuracy, save_directory, trial)
        tools.EvaluateModel(x_test, y_test, model, centroids, epoch,
                            trial, batch_size, True, False, save_directory)

        print("Saving test and training accuracy values")
        tools.SaveValues(train_pseudo_accuracy[-1].numpy(), test_pseudo_accuracy[-1].numpy(), trial, results_save_file)

        print("Plotting pair-wise centroid distances")
        tools.PlotAverageCentroidDistance(average_centroid_distance, save_directory, trial)

    print("=============================Done!=============================")






