import random

import numpy as np

def forward(X, W1, b1, W2, b2):
    # Step 1: Forward propagation to hidden layer (Z1 = W1 * X + b1)
    Z1 = np.dot(W1, X) + b1  # Z1 will have shape (hidden_layer_size, 60000)
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2  # Bias for output layer
    A2 = softmax(Z2)

    return A1, A2

def cross_entropy_loss(A2, labels, epsilon=1e-8):
    A2 = np.clip(A2, epsilon, 1. - epsilon)
    m = labels.shape[1]  # Number of samples
    loss = -np.sum(labels * np.log(A2)) / m  # Sum over all classes and samples plus L2 regularization
    return loss

def backpropagate(X, Y, A1, A2, W1, W2, lambda_):
    m = A2.shape[1]  # Number of training examples

    # Create array of correct size for one hot
    one_hot = np.zeros((10, Y.size))
    one_hot[Y, np.arange(Y.size)] = 1

    # Output layer gradients (softmax + cross-entropy)
    dZ2 = A2 - one_hot
    dW2 = np.dot(dZ2, A1.T) / m + (lambda_ * W2) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # Hidden layer gradients (ReLU backpropagation)
    dZ1 = np.dot(W2.T, dZ2) * (A1 > 0)  # Derivative of ReLU is 1 if > 0, else 0
    dW1 = np.dot(dZ1, X.T) / m + (lambda_ * W1) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2

def ReLU(Z):
    return np.maximum(0, Z) # Return 0 if less than 0, value unchanged otherwise

def softmax(Z):
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)

    exp_Z = np.exp(Z_stable)
    sum_exp_Z = np.sum(exp_Z, axis=0, keepdims=True)
    
    probabilities = exp_Z / sum_exp_Z
    return probabilities

def l2_regularization(W1, W2, lambda_):
    return lambda_ * (np.sum(W1**2) + np.sum(W2**2))

def gradient_descent(X, Y, W1, b1, W2, b2, learning_rate, num_epochs, show_predicted):
    A2 = None
    accuracy = []
    for epoch in range(num_epochs):
        A2, W1, b1, W2, b2, loss = single_pass(X, Y, W1, b1, W2, b2, learning_rate)

        # Print the loss every 100 epochs for monitoring
        if epoch % 50 == 0:
            # Calculate predicted value for each image
            y_hat = np.argmax(A2, axis=0)
            correct_count = 0

            if show_predicted:
                for i in range(5):
                    image_number = random.randint(0, len(y_hat))
                    print(f"{i+1}) Predicted: {y_hat[image_number]}, Actual: {Y[image_number]}")

            for i in range(len(y_hat)):
                if Y[i] == y_hat[i]:
                    correct_count += 1

            accuracy.append([learning_rate, epoch, round((correct_count/len(y_hat)) * 100, 2), round(loss, 2)])
            print(f"Learning Rate: {learning_rate}, Epoch: {epoch}, Accuracy: {round((correct_count/len(y_hat)) * 100, 2)}%, Loss: {round(loss,2)}")

    y_hat = np.argmax(A2, axis=0)
    correct_count = 0
    for i in range(len(y_hat)):
        if Y[i] == y_hat[i]:
            correct_count += 1
    return W1, b1, W2, b2, accuracy

def single_pass(X, Y, W1, b1, W2, b2, learning_rate):

    # Forward pass
    A1, A2 = forward(X, W1, b1, W2, b2)

    one_hot = np.zeros((10, Y.size))
    one_hot[Y, np.arange(Y.size)] = 1

    # Compute loss
    loss = (cross_entropy_loss(A2, one_hot, epsilon=1e-8) + l2_regularization(W1, W2, 1.0) / W1.shape[1]) * 100

    # Backpropagation
    dW1, db1, dW2, db2 = backpropagate(X, Y, A1, A2, W1, W2, 1.0)

    # Update weights and biases using gradient descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return A2, W1, b1, W2, b2, loss

def calculate_accuracy(y_hat, Y):
    correct_count = 0
    for i in range(len(y_hat)):
        if Y[i] == y_hat[i]:
            correct_count += 1

    return round(correct_count * 100 / len(y_hat), 2)