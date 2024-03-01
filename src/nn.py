import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from os.path import join

from dreader import MnistDataloader

DATA_PATH = "./data/mnist/"
TRAINING_SIZE = 60000  # max 60000

# paths
training_images_filepath = join(
    DATA_PATH, "train-images-idx3-ubyte/train-images-idx3-ubyte"
)
training_labels_filepath = join(
    DATA_PATH, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)
test_images_filepath = join(DATA_PATH, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
test_labels_filepath = join(DATA_PATH, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# get data as numpy array
x_train = np.array(x_train)[:TRAINING_SIZE] / 255
y_train = np.array(y_train)[:TRAINING_SIZE]
x_test = np.array(x_test) / 255
y_test = np.array(y_test)

input_size = x_train.shape[1] * x_train.shape[2]

# transform the data into 2D arrays with 28*28=784 pixels

x_train = x_train.reshape(x_train.shape[0], input_size)
x_test = x_test.reshape(x_test.shape[0], input_size)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


def init_params():
    W1 = np.random.rand(10, input_size) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x)  # we prevent overflow by subtracting the max
    return exp_x / exp_x.sum(axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, x):
    Z1 = W1.dot(x) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def one_hot(y):
    one_hot_y = np.zeros(
        (y.size, y.max() + 1)
    )  # create a matrix of y.size rows and y.max()+1 (10) columns
    one_hot_y[np.arange(y.size), y] = (
        1  # set the value of the column at the index of y to 1
    )
    one_hot_y = (
        one_hot_y.T
    )  # transpose the matrix bc we want each column to be an example

    return one_hot_y


def deriv_relu(x):
    return x > 0


def back_prop(Z1, A1, Z2, A2, W1, W2, x, y):
    one_hot_y = one_hot(y)
    dZ2 = A2 - one_hot_y
    dW2 = 1 / TRAINING_SIZE * dZ2.dot(A1.T)
    db2 = 1 / TRAINING_SIZE * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = 1 / TRAINING_SIZE * dZ1.dot(x.T)
    db1 = 1 / TRAINING_SIZE * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(A2, y):
    predictions = get_predictions(A2)
    return np.mean(predictions == y)


def get_loss(A2, y):
    one_hot_y = one_hot(y)
    # make sure we don't fail because of 0s in A2
    return -np.sum(one_hot_y * np.log(A2 + 1e-8)) / TRAINING_SIZE


def gradient_descent(x, y, iters, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iters):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x.T)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, x.T, y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(A2, y))
            print("Loss: ", get_loss(A2, y))

    return W1, b1, W2, b2


if __name__ == "__main__":
    W1, b1, W2, b2 = gradient_descent(x_train, y_train, 1000, 0.08)

    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x_test.T)
    print("Test accuracy: ", get_accuracy(A2, y_test))
    print("Test loss: ", get_loss(A2, y_test))
    print("Predictions: ", get_predictions(A2))
    print("True values: ", y_test)
