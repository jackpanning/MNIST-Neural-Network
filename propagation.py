import random

import numpy as np

def forward(X, W1, b1, W2, b2):
    """
    Perform forward propagation for a neural network.

    Parameters:
    - X: Input data of shape (784, 60000), where 784 is the input feature size and 60000 is the number of samples
    - W1: Weights for the first layer (hidden layer), shape (hidden_layer_size, 784)
    - b1: Biases for the first layer, shape (hidden_layer_size, 1)
    - W2: Weights for the second layer (output layer), shape (10, hidden_layer_size)
    - b2: Biases for the second layer (output layer), shape (10, 1)

    Returns:
    - A2: Final output layer activations, shape (10, 60000)
    """
    
    # Step 1: Forward propagation to hidden layer (Z1 = W1 * X + b1)
    Z1 = np.dot(W1, X) + b1  # Z1 will have shape (hidden_layer_size, 60000)
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2  # Bias for output layer
    A2 = softmax(Z2)

    return A1, A2

def cross_entropy_loss(A2, labels, epsilon=1e-8):
    """
    Compute loss for layer A2

    Parameters:
    - A2: Softmax output from the final layer of network
    - labels: True labels for each image from original dataset

    Returns:
    - loss: Cross-entropy loss
    """
    A2 = np.clip(A2, epsilon, 1. - epsilon)
    m = len(labels)
    loss = -np.sum(labels * np.log(A2)) / m

    return loss

def backpropagate(X, Y, A1, A2, W2):
    m = len(A2[0]) # Number of training examples

    # Create array of correct size for one hot
    one_hot = np.zeros((10, Y.size))

    # Set the appropriate element to 1 for each label
    one_hot[Y, np.arange(Y.size)] = 1

    # Output layer gradients (softmax + cross-entropy)
    dZ2 = A2 - one_hot
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # Hidden layer gradients (ReLU backpropagation)
    dZ1 = np.dot(W2.T, dZ2) * (A1 > 0)  # Derivative of ReLU is 1 if > 0, else 0
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2

def ReLU(Z):
    """
    ReLU activation function to convert any negative values to 0

    Parameters:
    - Z: Input data, logits of shape (10,60000)

    Returns:
    - Activations: Range of [0..inf], shape (10,60000)
    """

    activations = np.maximum(0, Z)
    
    return activations

def softmax(Z):
    """
    Softmax activation function to convert logits into probabilities.
    
    Parameters:
    - Z: Input data, logits, shape (10, 60000)
    
    Returns:
    - probabilities: Softmax output, shape (10, 60000)
    """

    Z_stable = Z - np.max(Z, axis=0, keepdims=True)

    exp_Z = np.exp(Z_stable)
    sum_exp_Z = np.sum(exp_Z, axis=0, keepdims=True)
    
    probabilities = exp_Z / sum_exp_Z
    return probabilities


def println(param):
    pass


def gradient_descent(X, Y, W1, b1, W2, b2, learning_rate, num_epochs, show_predicted):
    for epoch in range(num_epochs):
        
        # Forward pass
        A1, A2 = forward(X, W1, b1, W2, b2)

        # Compute loss
        loss = cross_entropy_loss(A2, Y)

        # Backpropagation
        dW1, db1, dW2, db2 = backpropagate(X, Y, A1, A2, W2)

        # Update weights and biases using gradient descent
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # Print the loss every 100 epochs for monitoring
        if epoch % 50 == 0:
            # Calculate predicted value for each image
            y_hat = np.argmax(A2, axis=0)
            correct_count = 0

            if show_predicted:
                for i in range(5):
                    image_number = random.randint(0, len(y_hat))
                    print(f"{i+1}) Predicted: {y_hat[image_number]}, Actual: {Y[image_number]}")

            print(f"Epoch {epoch}, Loss: {round(loss, 2)}")

            for i in range(len(y_hat)):
                if Y[i] == y_hat[i]:
                    correct_count += 1
            accuracy = round((correct_count/len(y_hat)) * 100, 2)
            print(f"Accuracy: {accuracy}%")


    return W1, b1, W2, b2