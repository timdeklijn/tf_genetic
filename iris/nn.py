import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class NN(tf.Module):
    def __init__(self, name=None):
        super(NN, self).__init__(name=name)

    def __call__(self, x, w, b):
        x = tf.nn.relu(x @ w[0] + b[0])
        return tf.nn.sigmoid(x @ w[1] + b[1])


def load_data():
    """
    Load and preprocess data
    """
    # Download data
    X, y = load_iris(return_X_y=True)
    # Convert y to categorical
    y = tf.keras.utils.to_categorical(y, num_classes=3)
    # Split and convert to tensors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    X_train = tf.cast(X_train, dtype=tf.float32)
    y_train = tf.cast(y_train, dtype=tf.float32)
    X_test = tf.cast(X_test, dtype=tf.float32)
    y_test = tf.cast(y_test, dtype=tf.float32)
    return X_train, X_test, y_train, y_test


def initialize_population(population_size, weights_shapes, bias_shapes):
    """
    Create two lists with weights and biases
    """
    ws, bs = [], []
    for _ in range(0, population_size):
        ws.append([tf.random.uniform(s, dtype=tf.float32) for s in weights_shapes])
        bs.append([tf.random.uniform(s, dtype=tf.float32) for s in bias_shapes])
    return ws, bs


def calc_fitness(ws, bs, nn, X_train, y_train):
    """
    Calculate fitness for all individuals in the population
    """
    scores = []
    for w, b in zip(ws, bs):
        predictions = nn(X_train, w, b)
        scores.append(
            tf.reduce_mean(tf.cast(y_train == predictions, dtype=tf.float32)).numpy()
        )
    return tf.cast(scores, dtype=tf.float32)


@tf.function
def get_parent_indices(population_fitness):
    """
    Basically np.random.coice using probabilities. Reshape this to a tensor
    with shape (len(population_fitness), 2).
    """
    # from: https://stackoverflow.com/questions/41123879/numpy-random-choice-in-tensorflow
    cum_dist = tf.cast(tf.math.cumsum(population_fitness), dtype=tf.float32)
    cum_dist /= cum_dist[-1]
    unif_samp = tf.random.uniform((2 * population_fitness.shape[0],), 0, 1)
    return tf.reshape(
        tf.searchsorted(cum_dist, unif_samp), (population_fitness.shape[0], 2)
    )


if __name__ == "__main__":
    # Set shapes of the weights and biases needed to run the NN
    # on the iris dataset
    weights_shapes = [(4, 10), (10, 3)]
    bias_shapes = [(10,), (3,)]
    population_size = 100

    X_train, X_test, y_train, y_test = load_data()
    nn = NN()
    ws, bs = initialize_population(population_size, weights_shapes, bias_shapes)
    fitness = calc_fitness(ws, bs, nn, X_train, y_train)
    parent_indices = get_parent_indices(fitness)
    print(parent_indices)
