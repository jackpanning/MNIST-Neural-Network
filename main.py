from mnist_sort import Sorter
from mnist_sort import initialize_weights

if __name__ == "__main__":
    mnist_network = Sorter()    
    W1, b1, W2, b2 = initialize_weights(hidden_layers=10)
    mnist_network.gradient_descent(W1, b1, W2, b2,learning_rate=0.01, num_epochs=501, show_predicted=True)