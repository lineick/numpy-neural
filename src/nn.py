import numpy as np

from os.path import join

from dreader import MnistDataloader

DATA_PATH = "./data/mnist/"
TRAINING_SIZE = 60000  # max 60000
STORE = True

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
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # prevent overflow
    return exp_x / exp_x.sum(axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, x):
    Z1 = W1.dot(x) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((y.size, num_classes))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def deriv_relu(x):
    return x > 0


def back_prop(Z1, A1, Z2, A2, W1, W2, x, y, batch_size):
    one_hot_y = one_hot(y, num_classes=10)
    dZ2 = A2 - one_hot_y
    dW2 = 1 / batch_size * dZ2.dot(A1.T)
    db2 = 1 / batch_size * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
    dW1 = 1 / batch_size * dZ1.dot(x.T)
    db1 = 1 / batch_size * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


def update_params_momentum(
    W1, b1, W2, b2, dW1, db1, dW2, db2, v_dW1, v_db1, v_dW2, v_db2, alpha, beta
):
    v_dW1 = beta * v_dW1 + (1 - beta) * dW1
    v_db1 = beta * v_db1 + (1 - beta) * db1
    v_dW2 = beta * v_dW2 + (1 - beta) * dW2
    v_db2 = beta * v_db2 + (1 - beta) * db2

    W1 = W1 - alpha * v_dW1
    b1 = b1 - alpha * v_db1
    W2 = W2 - alpha * v_dW2
    b2 = b2 - alpha * v_db2

    return W1, b1, W2, b2, v_dW1, v_db1, v_dW2, v_db2


def update_params_adam(
    W1,
    b1,
    W2,
    b2,
    dW1,
    db1,
    dW2,
    db2,
    m_W1,
    v_W1,
    m_b1,
    v_b1,
    m_W2,
    v_W2,
    m_b2,
    v_b2,
    t,
    alpha,
    beta1,
    beta2,
    epsilon,
):
    # Update biased first moment estimate
    m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
    m_b1 = beta1 * m_b1 + (1 - beta1) * db1
    m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
    m_b2 = beta1 * m_b2 + (1 - beta1) * db2

    # Update biased second raw moment estimate
    v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1**2)
    v_b1 = beta2 * v_b1 + (1 - beta2) * (db1**2)
    v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2**2)
    v_b2 = beta2 * v_b2 + (1 - beta2) * (db2**2)

    # Compute bias-corrected first moment estimate
    m_W1_corr = m_W1 / (1 - beta1**t)
    m_b1_corr = m_b1 / (1 - beta1**t)
    m_W2_corr = m_W2 / (1 - beta1**t)
    m_b2_corr = m_b2 / (1 - beta1**t)

    # Compute bias-corrected second raw moment estimate
    v_W1_corr = v_W1 / (1 - beta2**t)
    v_b1_corr = v_b1 / (1 - beta2**t)
    v_W2_corr = v_W2 / (1 - beta2**t)
    v_b2_corr = v_b2 / (1 - beta2**t)

    # Update parameters
    W1 = W1 - alpha * m_W1_corr / (np.sqrt(v_W1_corr) + epsilon)
    b1 = b1 - alpha * m_b1_corr / (np.sqrt(v_b1_corr) + epsilon)
    W2 = W2 - alpha * m_W2_corr / (np.sqrt(v_W2_corr) + epsilon)
    b2 = b2 - alpha * m_b2_corr / (np.sqrt(v_b2_corr) + epsilon)

    return W1, b1, W2, b2, m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(A2, y):
    predictions = get_predictions(A2)
    return np.mean(predictions == y)


def get_loss(A2, y):
    one_hot_y = one_hot(y, num_classes=10)
    # make sure we don't fail because of 0s in A2
    return -np.sum(one_hot_y * np.log(A2 + 1e-8)) / y.size


def gradient_descent(
    x,
    y,
    iters,
    alpha,
    batch_size,
    optimizer="sgd",
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    W1, b1, W2, b2 = init_params()
    n_samples = x.shape[0]

    if optimizer == "momentum":
        v_dW1 = np.zeros_like(W1)
        v_db1 = np.zeros_like(b1)
        v_dW2 = np.zeros_like(W2)
        v_db2 = np.zeros_like(b2)
        beta = beta1  # use beta1 for momentum
    elif optimizer == "adam":
        m_W1 = np.zeros_like(W1)
        v_W1 = np.zeros_like(W1)
        m_b1 = np.zeros_like(b1)
        v_b1 = np.zeros_like(b1)
        m_W2 = np.zeros_like(W2)
        v_W2 = np.zeros_like(W2)
        m_b2 = np.zeros_like(b2)
        v_b2 = np.zeros_like(b2)
        t = 0

    for i in range(iters):
        permutation = np.random.permutation(n_samples)
        x_shuffled = x[permutation]
        y_shuffled = y[permutation]
        for j in range(0, n_samples, batch_size):
            if optimizer == "adam":
                t += 1  # time step
            x_batch = x_shuffled[j : j + batch_size]
            y_batch = y_shuffled[j : j + batch_size]
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x_batch.T)
            dW1, db1, dW2, db2 = back_prop(
                Z1, A1, Z2, A2, W1, W2, x_batch.T, y_batch, x_batch.shape[0]
            )

            if optimizer == "sgd":
                W1, b1, W2, b2 = update_params(
                    W1, b1, W2, b2, dW1, db1, dW2, db2, alpha
                )
            elif optimizer == "momentum":
                W1, b1, W2, b2, v_dW1, v_db1, v_dW2, v_db2 = update_params_momentum(
                    W1,
                    b1,
                    W2,
                    b2,
                    dW1,
                    db1,
                    dW2,
                    db2,
                    v_dW1,
                    v_db1,
                    v_dW2,
                    v_db2,
                    alpha,
                    beta,
                )
            elif optimizer == "adam":
                W1, b1, W2, b2, m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2 = (
                    update_params_adam(
                        W1,
                        b1,
                        W2,
                        b2,
                        dW1,
                        db1,
                        dW2,
                        db2,
                        m_W1,
                        v_W1,
                        m_b1,
                        v_b1,
                        m_W2,
                        v_W2,
                        m_b2,
                        v_b2,
                        t,
                        alpha,
                        beta1,
                        beta2,
                        epsilon,
                    )
                )
        if i % 10 == 0:
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x.T)
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(A2, y))
            print("Loss: ", get_loss(A2, y))
    return W1, b1, W2, b2


def save_model(W1, b1, W2, b2, path):
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Model saved to {path}")


def load_model(path):
    data = np.load(path)
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    print(f"Model loaded from {path}")
    return W1, b1, W2, b2


if __name__ == "__main__":
    # Hyperparameters
    iters = 150  # Number of epochs
    alpha = 0.001  # Learning rate
    batch_size = 32  # Mini-batch size
    optimizer = "adam"  # Choose 'sgd', 'momentum', or 'adam'

    # Train the model
    W1, b1, W2, b2 = gradient_descent(
        x_train, y_train, iters, alpha, batch_size, optimizer=optimizer
    )

    # Evaluate on test data
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x_test.T)
    accuracy = get_accuracy(A2, y_test)
    print("Test accuracy: ", accuracy)
    print("Test loss: ", get_loss(A2, y_test))
    print("Predictions: ", get_predictions(A2))
    print("True values: ", y_test)

    # Specify the path where you want to save the model
    MODEL_PATH = (
        DATA_PATH + f"model_{str(accuracy)}_{optimizer}.npz"
    )  # You can change this to your desired path

    if STORE:
        # Save the model parameters
        save_model(W1, b1, W2, b2, MODEL_PATH)
