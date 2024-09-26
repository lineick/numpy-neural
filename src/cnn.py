import numpy as np
from os.path import join

from dreader import MnistDataloader

DATA_PATH = "./data/mnist/"
TRAINING_SIZE = 60000  # max 60000
STORE = True

# Initialize parameters
# Conv layer parameters
num_filters = 16
filter_size = 3  # 3x3 filters
channels = 1  # grey scale
fan_in = channels * filter_size * filter_size  # channels = 1
W1 = np.random.randn(num_filters, 1, filter_size, filter_size) * np.sqrt(2.0 / fan_in)
b1 = np.zeros(num_filters)

# Fully connected layer parameters
fc_input_dim = num_filters * 14 * 14  # Corrected calculation
num_classes = 10
fan_in_fc = fc_input_dim  # Number of input units
W2 = np.random.randn(fc_input_dim, num_classes) * np.sqrt(2.0 / fan_in_fc)
b2 = np.zeros((1, num_classes))

# Initialize Adam parameters
m_W1 = np.zeros_like(W1)
v_W1 = np.zeros_like(W1)
m_b1 = np.zeros_like(b1)
v_b1 = np.zeros_like(b1)

m_W2 = np.zeros_like(W2)
v_W2 = np.zeros_like(W2)
m_b2 = np.zeros_like(b2)
v_b2 = np.zeros_like(b2)


def conv_forward(X, W, b, stride=1, padding=0):
    batch_size, channels, height, width = X.shape
    num_filters, _, filter_height, filter_width = W.shape

    # Output dimensions
    out_height = int((height - filter_height + 2 * padding) / stride) + 1
    out_width = int((width - filter_width + 2 * padding) / stride) + 1

    # Pad input
    X_padded = np.pad(
        X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )

    # Extract patches
    k = filter_height * filter_width * channels
    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    d = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

    X_col = X_padded[:, d, i, j]
    X_col = X_col.transpose(1, 2, 0).reshape(k, -1)

    # Reshape W
    W_col = W.reshape(num_filters, -1)

    # Perform convolution
    out = W_col @ X_col + b.reshape(-1, 1)
    out = out.reshape(num_filters, out_height, out_width, batch_size)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col, W_col)
    return out, cache


def relu_forward(X):
    out = np.maximum(0, X)
    cache = X
    return out, cache


def maxpool_forward(X, pool_height, pool_width, stride):
    batch_size, channels, height, width = X.shape
    out_height = int((height - pool_height) / stride) + 1
    out_width = int((width - pool_width) / stride) + 1

    X_reshaped = X.reshape(batch_size * channels, 1, height, width)

    X_col = np.lib.stride_tricks.as_strided(
        X_reshaped,
        shape=(
            batch_size * channels,
            1,
            out_height,
            out_width,
            pool_height,
            pool_width,
        ),
        strides=(
            X_reshaped.strides[0],
            X_reshaped.strides[1],
            stride * X_reshaped.strides[2],
            stride * X_reshaped.strides[3],
            X_reshaped.strides[2],
            X_reshaped.strides[3],
        ),
        writeable=False,
    )

    X_col = X_col.reshape(
        batch_size * channels, out_height * out_width, pool_height * pool_width
    )

    out = np.max(X_col, axis=2)
    out = out.reshape(batch_size, channels, out_height, out_width)

    cache = (X, pool_height, pool_width, stride, X_col, out)
    return out, cache


def flatten_forward(X):
    batch_size = X.shape[0]
    out = X.reshape(batch_size, -1)
    cache = X.shape
    return out, cache


def fc_forward(X, W, b):
    out = X.dot(W) + b
    cache = (X, W, b)
    return out, cache


def softmax_forward(X):
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))  # Stability trick
    out = exps / np.sum(exps, axis=1, keepdims=True)
    cache = out
    return out, cache


def cross_entropy_loss(preds, labels):
    batch_size = preds.shape[0]
    # Convert labels to one-hot encoding
    one_hot_labels = np.zeros_like(preds)
    one_hot_labels[np.arange(batch_size), labels] = 1

    # Compute loss
    loss = -np.sum(one_hot_labels * np.log(preds + 1e-8)) / batch_size

    # Compute gradient
    dX = (preds - one_hot_labels) / batch_size
    return loss, dX


def fc_backward(dout, cache):
    X, W, b = cache
    dX = dout.dot(W.T)
    dW = X.T.dot(dout)
    db = np.sum(dout, axis=0, keepdims=True)
    return dX, dW, db


def flatten_backward(dout, original_shape):
    dX = dout.reshape(original_shape)
    return dX


def maxpool_backward(dout, cache):
    X, pool_height, pool_width, stride, X_col, out = cache
    batch_size, channels, height, width = X.shape
    out_height, out_width = dout.shape[2], dout.shape[3]

    dX_col = np.zeros_like(X_col)

    dout_flat = dout.transpose(0, 2, 3, 1).ravel()

    idx = np.argmax(X_col, axis=2).flatten()

    dX_col.reshape(-1, X_col.shape[2])[np.arange(len(idx)), idx] = dout_flat

    dX_col = dX_col.reshape(
        batch_size * channels, out_height, out_width, pool_height, pool_width
    )
    dX_col = dX_col.transpose(0, 1, 2, 3, 4)

    dX = np.zeros((batch_size * channels, height, width))

    for h in range(out_height):
        for w in range(out_width):
            h_start = h * stride
            h_end = h_start + pool_height
            w_start = w * stride
            w_end = w_start + pool_width
            dX[:, h_start:h_end, w_start:w_end] += dX_col[:, h, w, :, :].reshape(
                batch_size * channels, pool_height, pool_width
            )

    dX = dX.reshape(batch_size, channels, height, width)

    return dX


def relu_backward(dout, cache):
    X = cache
    dX = dout.copy()
    dX[X <= 0] = 0
    return dX


def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col, W_col = cache
    batch_size, channels, height, width = X.shape
    num_filters, _, filter_height, filter_width = W.shape
    batch_size, num_filters, out_height, out_width = dout.shape

    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, num_filters)

    # Gradients with respect to weights
    dW_col = dout_flat.T.dot(X_col.T)

    dW = dW_col.reshape(num_filters, channels, filter_height, filter_width)

    # Gradients with respect to bias
    db = np.sum(dout, axis=(0, 2, 3))

    # Gradients with respect to input
    dX_col = W_col.T.dot(dout_flat.T)

    # Reshape dX_col to match the shape of im2col indices
    dX_col = dX_col.T
    k = channels * filter_height * filter_width
    dX_col = dX_col.reshape(batch_size, out_height * out_width, k)
    dX_col = dX_col.transpose(0, 2, 1)
    dX_col = dX_col.reshape(batch_size * k, out_height * out_width)

    # Reconstruct the input image gradients
    dX_padded = np.zeros(
        (batch_size, channels, height + 2 * padding, width + 2 * padding)
    )

    # Reconstruct dX from dX_col
    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    d = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

    np.add.at(
        dX_padded,
        (slice(None), d, i, j),
        dX_col.reshape(
            batch_size, channels * filter_height * filter_width, out_height * out_width
        ),
    )

    # Remove padding
    if padding > 0:
        dX = dX_padded[:, :, padding:-padding, padding:-padding]
    else:
        dX = dX_padded

    return dX, dW, db


def train_cnn(
    x_train, y_train, x_test, y_test, epochs=3, learning_rate=0.001, batch_size=128
):
    global W1, b1, W2, b2, m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2  # Include Adam parameters

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    t = 0  # Time step

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # Shuffle training data
        permutation = np.random.permutation(x_train.shape[0])
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]

        num_batches = x_train.shape[0] // batch_size

        for i in range(num_batches):
            t += 1  # Increment time step
            X_batch = x_train_shuffled[i * batch_size : (i + 1) * batch_size]
            y_batch = y_train_shuffled[i * batch_size : (i + 1) * batch_size]

            # Forward pass
            conv_out, conv_cache = conv_forward(X_batch, W1, b1, stride=1, padding=1)
            relu_out, relu_cache = relu_forward(conv_out)
            pool_out, pool_cache = maxpool_forward(
                relu_out, pool_height=2, pool_width=2, stride=2
            )
            flatten_out, flatten_cache = flatten_forward(pool_out)
            fc_out, fc_cache = fc_forward(flatten_out, W2, b2)
            probs, softmax_cache = softmax_forward(fc_out)

            # Compute loss and initial gradient
            loss, dloss = cross_entropy_loss(probs, y_batch)

            # Backward pass
            dflatten, dW2, db2 = fc_backward(dloss, fc_cache)
            dpool = flatten_backward(dflatten, flatten_cache)
            drelu = maxpool_backward(dpool, pool_cache)
            dconv = relu_backward(drelu, relu_cache)
            dX, dW1, db1 = conv_backward(dconv, conv_cache)

            # Adam update for W1 and b1
            m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
            v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1**2)
            m_hat_W1 = m_W1 / (1 - beta1**t)
            v_hat_W1 = v_W1 / (1 - beta2**t)
            W1 -= learning_rate * m_hat_W1 / (np.sqrt(v_hat_W1) + epsilon)

            m_b1 = beta1 * m_b1 + (1 - beta1) * db1
            v_b1 = beta2 * v_b1 + (1 - beta2) * (db1**2)
            m_hat_b1 = m_b1 / (1 - beta1**t)
            v_hat_b1 = v_b1 / (1 - beta2**t)
            b1 -= learning_rate * m_hat_b1 / (np.sqrt(v_hat_b1) + epsilon)

            # Adam update for W2 and b2
            m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
            v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2**2)
            m_hat_W2 = m_W2 / (1 - beta1**t)
            v_hat_W2 = v_W2 / (1 - beta2**t)
            W2 -= learning_rate * m_hat_W2 / (np.sqrt(v_hat_W2) + epsilon)

            m_b2 = beta1 * m_b2 + (1 - beta1) * db2
            v_b2 = beta2 * v_b2 + (1 - beta2) * (db2**2)
            m_hat_b2 = m_b2 / (1 - beta1**t)
            v_hat_b2 = v_b2 / (1 - beta2**t)
            b2 -= learning_rate * m_hat_b2 / (np.sqrt(v_hat_b2) + epsilon)

            if i % 10 == 0:
                print(f"Batch {i}/{num_batches}, Loss: {loss:.4f}")

        # Evaluate on training data
        train_accuracy = evaluate_cnn(
            x_train[:1000], y_train[:1000]
        )  # Use a subset for speed
        print(f"Training Accuracy after epoch {epoch+1}: {train_accuracy:.4f}")

        # Evaluate on test data
        test_accuracy = evaluate_cnn(x_test, y_test)
        print(f"Test Accuracy after epoch {epoch+1}: {test_accuracy:.4f}")


def evaluate_cnn(X, y):
    # Forward pass
    conv_out, _ = conv_forward(X, W1, b1, stride=1, padding=1)  # Updated padding
    relu_out, _ = relu_forward(conv_out)
    pool_out, _ = maxpool_forward(relu_out, pool_height=2, pool_width=2, stride=2)
    flatten_out, _ = flatten_forward(pool_out)
    fc_out, _ = fc_forward(flatten_out, W2, b2)
    probs, _ = softmax_forward(fc_out)

    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy


def save_model(filepath):
    """Saves the model parameters to a specified file."""
    np.savez(filepath, W1=W1, b1=b1, W2=W2, b2=b2)


def load_model(filepath):
    """Loads the model parameters from a specified file."""
    global W1, b1, W2, b2
    data = np.load(filepath)
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]


if __name__ == "__main__":
    # paths
    training_images_filepath = join(
        DATA_PATH, "train-images-idx3-ubyte/train-images-idx3-ubyte"
    )
    training_labels_filepath = join(
        DATA_PATH, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
    )
    test_images_filepath = join(
        DATA_PATH, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    )
    test_labels_filepath = join(
        DATA_PATH, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
    )

    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # get data as numpy array
    x_train = np.array(x_train)[:TRAINING_SIZE] / 255.0
    y_train = np.array(y_train)[:TRAINING_SIZE]
    x_test = np.array(x_test) / 255.0
    y_test = np.array(y_test)

    input_size = x_train.shape[1] * x_train.shape[2]

    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    train_cnn(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=5,
        learning_rate=0.0015,
        batch_size=100,
    )

    model_save_path = f"../data/mnist/cnn_{evaluate_cnn(x_test, y_test)}.npz"

    if model_save_path is not None:
        save_model(model_save_path)
        print(f"Model saved to {model_save_path}")
