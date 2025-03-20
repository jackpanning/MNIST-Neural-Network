# MNIST Sorter
This is a classification neural network written from scratch using the Modified National Institute of Standards and
Technology database. By implementing gradient descent, I created a straightforward architecture capable of achieving
over 90% accuracy.

## How It's Made:

**Tech used:** Python, tkinter, numpy, matplotlib

**Network Architecture:**
The MNIST dataset includes 60,000 train images and 10,000 test images, each depicting a handwritten numerical digit at a 28x28 pixel resolution. Since all images are grayscale and anti-aliased, the image can be represented as a 28x28 numpy array, which is flatted to 784, giving us the 60,000x784 dimension of the input layer.

For forward propagation, the network has multiple hidden layers, each with a customizable amount of nodes. Weights are generating using He initialization, which in combination with ReLU helps to maintain optimal gradient flow. The last layer has length 10, corresponding to digits 0 through 9. Finally, a softmax function is applied to give relative probabilities.

Backpropagation takes the corresponding labels for each training image and one-hot encodes the value, giving an array of the same dimension as the final layer. Cross-entropy loss is calculated as the difference between the predicted probabilities of digits and the actual distribution. L2 regularization is also applied in order to prevent overfitting to train data. Then, weights are adjusted based on the hyperparameter learning rate.
This process repeats over a specified number of epochs, with loss and accuracy calculated for each.

**GUI:**
The GUI was implemented using the tkinter framework, giving the user a convenient interface to guage model performance, change parameters, log past runs, and see a visualization of the network. The tkinter framework proved effortless to use, with the majority of elements being implemented as classes, meaning I can easily reuse code for future projects.

## Lessons Learned:

This project proved to be eye-opening experience in the field of machine learning. For a long time, machine learning has mystified myself and others, but once I sat down and dove into the admittedly straightforward math, I learned a great deal about gradients, forward and back propagation, and the tuning of hyperparameters. Compared to the cutting edge of AI, this is a simple network architecture, but these topics will remain relevant as I learn more.

The tkinter framework, used when designing the gui, taught me lots about front-end development. Documentation online proved to be comprehensive, which taught me everything I needed about widgets, parameters, and how to extend functionality to create my own widget classes. 
