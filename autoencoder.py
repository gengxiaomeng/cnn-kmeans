import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle


TRAIN_BUF = 60000
BATCH_SIZE = 100

TEST_BUF = 10000

def  CreateAutoEncoder():

    encoder_input = keras.Input(shape=(28, 28, 1), name='img')
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    encoder_output = layers.Conv2D(8, 3, padding='same', activation='relu')(x)
   
    encoder = keras.Model(encoder_input, encoder_output, name='encoder')
    encoder.summary()

    x = layers.Conv2DTranspose(8, 3, activation='relu', padding='same')(encoder_output)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(16, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoder_output = layers.Conv2DTranspose(1, 3, activation='relu', padding='same')(x)
    
    autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
    autoencoder.summary()
    return autoencoder, encoder

if __name__ == "__main__":
    
    tf.keras.backend.clear_session()

    (train_images, _), (test_images, y_test) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    autoencoder, encoder = CreateAutoEncoder()
    
    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    autoencoder.compile(optimizer=optimizer, loss="mse")
    
    autoencoder.summary()

    history = autoencoder.fit(train_images, train_images, epochs=10, batch_size=64, verbose=1)

    restored_testing_dataset = autoencoder.predict(test_images)
    
    plt.figure(figsize=(20,5))
    
    for i in range(10):
        index = y_test.tolist().index(i)
        plt.subplot(2, 10, i+1)
        plt.imshow(test_images[index].reshape((28,28)))
        plt.gray()
        plt.subplot(2, 10, i+11)
        plt.imshow(restored_testing_dataset[index].reshape((28,28)))
        plt.gray()
        
