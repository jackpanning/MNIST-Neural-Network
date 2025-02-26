import random
import numpy as np # linear algebra
import struct
import matplotlib.pyplot as plt

import propagation

# Change as desired
def initialize_weights(hidden_layers=10):
    W1 = np.random.randn(hidden_layers,784) * np.sqrt(2. / 784) # 784 is 28^2, or 2d array transformed to 1d
    b1 = np.zeros((hidden_layers,1))
    W2 = np.random.randn(10,hidden_layers)
    b2 = np.zeros((10,1))
    return W1, b1, W2, b2

class Sorter:
    def __init__(self):

        # Variables
        self._learning_rate =  0.005
        self._hidden_layers = 128
        self._epochs = 10

        # Load files
        training_images_filepath = 'dataset/train-images-idx3-ubyte'
        training_labels_filepath = 'dataset/train-labels-idx1-ubyte'
        test_images_filepath = 'dataset/t10k-images-idx3-ubyte'
        test_labels_filepath = 'dataset/t10k-labels-idx1-ubyte'

        # Opens file to read using binary 
        with open(training_images_filepath, 'rb') as f:
            # Validate that file is actually an MNIST data file
            magic_number = struct.unpack('>I', f.read(4))[0]
            if magic_number != 2051:
                raise ValueError(f"Expected magic number 2051 for image, got: {magic_number}")
            
            # Read data from .ubyte file
            image_index, rows, cols = struct.unpack('>III', f.read(12))

            train_images = np.frombuffer(f.read(), dtype=np.uint8)
            train_images = train_images.reshape(image_index, rows, cols)  # Reshape into (image_index, 28, 28)
            train_images = train_images.astype(np.float32) / 255 # Normalize values from [0..255] to [0..1]

        with open(training_labels_filepath, 'rb') as f:
            # Validate that file is actually an MNIST data file
            magic_number = struct.unpack('>I', f.read(4))[0]
            if magic_number != 2049:
                raise ValueError(f"Expected magic number 2051 for image, got: {magic_number}")

            struct.unpack('>I', f.read(4))
            self._input_labels = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape from 60000x28x28 into 60000x784 and 2: transpose into 784x600000
        train_images = train_images.reshape(60000, -1)
        train_images = train_images.transpose()
        self._input_layer = train_images
        self._weights = initialize_weights(hidden_layers=100)

    def gradient_descent(self, W1, b1, W2, b2, learning_rate, num_epochs, show_predicted=False):
        print(self._input_labels.shape)
        W1, b1, W2, b2 = propagation.gradient_descent(self._input_layer, self._input_labels, W1, b1, W2, b2, learning_rate, num_epochs, show_predicted)
        return W1, b1, W2, b2

    def test_weights(input_layer,W1, b1, W2, b2):
        A1, A2 = propagation.forward(input_layer,W1,b1,W2,b2)
        return A2

    def get_input(self):
        return self._input_layer