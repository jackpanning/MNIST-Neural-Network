import numpy as np # linear algebra
import struct

import propagation
from propagation import gradient_descent

# Change as desired
def initialize_weights(hidden_layers):
    W1 = np.random.randn(hidden_layers,784) * np.sqrt(2. / 784) # 784 is 28^2, or 2d array transformed to 1d
    b1 = np.zeros((hidden_layers,1))
    W2 = np.random.randn(10,hidden_layers)
    b2 = np.zeros((10,1))
    return W1, b1, W2, b2

class Sorter:
    def __init__(self, learning_rate=0.005, hidden_layers=32, epochs=100):

        # Variables
        self._learning_rate = learning_rate
        self._hidden_layers = hidden_layers
        self._epochs = epochs

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
        self.W1, self.b1, self.W2, self.b2 = initialize_weights(hidden_layers=self._hidden_layers)
        self.A2 = np.zeros((10,1))

    def gradient_descent(self, W1, b1, W2, b2, learning_rate, num_epochs, show_predicted=False):
        W1, b1, W2, b2, accuracy = propagation.gradient_descent(self._input_layer, self._input_labels, W1, b1, W2, b2, learning_rate, num_epochs, show_predicted)
        return W1, b1, W2, b2, accuracy

    def gui_gradient_descent(self, gui, progress, num_epochs, learning_rate, hidden_layers):
        self.W1, self.b1, self.W2, self.b2 = initialize_weights(hidden_layers=hidden_layers)
        gui.visualization.nodes_per_layer[1] = hidden_layers
        gui.visualization.create_visualization()

        self._epochs = num_epochs
        self._learning_rate = learning_rate
        loss = 0
        train_accuracy = 0
        validation_accuracy = 0
        progress.progress = 0
        factor = 100 / self._epochs
        X_val, Y_val = self.create_validation_set(self._input_layer, self._input_labels, 0.2)

        for i in range(self._epochs):
            self.A2, self.W1, self.b1, self.W2, self.b2, loss = (
                propagation.single_pass(self._input_layer, self._input_labels, self.W1, self.b1, self.W2,
                                        self.b2, self._learning_rate))
            print(round(loss, 2))
            if i * factor > progress.progress:
                progress.progress += 1 if factor < 1 else int(factor)
                progress.progress_bar.update_progress(progress.progress)
                gui.update_gui()

            y_hat = np.argmax(self.A2, axis=0)
            train_accuracy = propagation.calculate_accuracy(self._input_labels, y_hat)

            _, A2_val = propagation.forward(X_val, self.W1, self.b1, self.W2, self.b2)
            y_hat_val = np.argmax(self.A2, axis=0)
            validation_accuracy = propagation.calculate_accuracy(Y_val, y_hat_val)

            progress.set_metrics(i + 1, loss, train_accuracy, validation_accuracy)

            if i % 10 == 0:
                print(f"Epoch: {i}\n" +
                      "Train: " + str(round(train_accuracy, 2)) + "%\n" +
                      "Validation: " + str(round(validation_accuracy, 2)) + "%")

        new_entry = [str(len(gui.results.results_arr) + 1),
                     f"Epochs: {self._epochs}\n"
                     f"Learning Rate: {self._learning_rate}\n"
                     f"Hidden Layer Nodes: {self._hidden_layers}",
                     str(round(loss, 2)),
                     str(round(train_accuracy, 2)) + "%",
                     str(round(validation_accuracy, 2)) + "%"]
        gui.results.results_arr.append(new_entry)
        gui.results.draw_entries()
        progress.progress = 100
        progress.progress_bar.update_progress(progress.progress)
        gui.update_gui()
        gui.thread = None
        return

    def test_weights(input_layer,W1, b1, W2, b2):
        A1, A2 = propagation.forward(input_layer,W1,b1,W2,b2)
        return A2

    def get_input(self):
        return self._input_layer

    def show_image(self, index):
        image = self._input_layer.transpose()
        image = image[index].reshape(28, 28)
        i = 0
        j = 0
        while i < 28:
            while j < 28:
                print(" " if image[i, j] == 0 else "1 ", end="")
                j = j + 1
            i = i + 1
            print()
            j = 0
        print(f"\nLabel: {self._input_labels[index]}")

    def test_rates(self, epochs, learning_rates):
        accuracies = []
        for lr in learning_rates:
            W1, b1, W2, b2 = initialize_weights(hidden_layers=64)
            W1, b1, W2, b2, accuracy = gradient_descent(self._input_layer, self._input_labels, W1, b1, W2, b2, lr, epochs, False)
            print(accuracy)
            for i in range(len(accuracy)):
                accuracies.append(accuracy[i])
        return accuracies

    def create_validation_set(self, X, Y, validation_split=0.2):
        # Shuffle the data
        indices = np.arange(X.shape[1])
        np.random.shuffle(indices)

        X_shuffled = X[:, indices]
        Y_shuffled = Y[indices]

        # Split the data
        split_index = int(X.shape[1] * (1 - validation_split))
        X_val = X_shuffled[:, split_index:]
        Y_val = Y_shuffled[split_index:]

        return X_val, Y_val