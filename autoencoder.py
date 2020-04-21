import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle


def CreateAutoEncoder():

    encoder_input = keras.Input(shape=(28, 28, 1), name='img')
    x = layers.Conv2D(300, 3, activation='relu', padding='same')(encoder_input)
    x = layers.Conv2D(300, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(200, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(200, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(100, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(100, 3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4000)(x)
    x = layers.Dense(2000)(x)
    x = layers.Dense(1000)(x)
    encoder_output = layers.Dense(500)(x)

    encoder = keras.Model(encoder_input, encoder_output, name='encoder')

    x = layers.Dense(1000)(encoder_output)
    x = layers.Dense(2000)(x)
    x = layers.Dense(4000)(x)
    x = layers.Dense(4900)(x)
    x = layers.Reshape(target_shape = (7,7,100))(x)
    x = layers.Conv2DTranspose(100, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(100, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(200, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(200, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(300, 3, activation='relu', padding='same')(x)
    decoder_output = layers.Conv2DTranspose(1, 3, activation='relu', padding='same')(x)

    autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')

    return autoencoder, encoder

def TrainAutoencoder(train_images, test_images, y_test, batch_size, epochs):

    # Shuffle training and test sets
    train_images = shuffle(train_images)
    test_images, y_test = shuffle(test_images, y_test)

    # Reshape training set to match network inputs
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

    # Grab encoder and autoencoder networks
    autoencoder, encoder = CreateAutoEncoder()

    # Print encoder and autoencoder summary
    encoder.summary()
    autoencoder.summary()

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(1e-4)

    # Compile autoencoder
    autoencoder.compile(optimizer=optimizer, loss="mse")

    # Train autoencoder
    history = autoencoder.fit(train_images, train_images, epochs=epochs, batch_size=batch_size, verbose=1)

    # Predict on test set
    restored_testing_dataset = autoencoder.predict(test_images)

    plt.figure(figsize=(20,5))

    for i in range(10):
        index = y_test.tolist().index(i)
        ax = plt.subplot(2, 10, i+1)
        plt.imshow(test_images[index].reshape((28,28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, 10, i+11)
        plt.imshow(restored_testing_dataset[index].reshape((28,28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
    # Save encoder model
    encoder.save("encoder_model")

    return encoder
