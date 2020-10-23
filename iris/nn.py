import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mut_range = 0.005
mut_prob = 0.01


def weigths_from_keras(X_train, y_train):
    model = tf.keras.Sequential()
    # Y should be categorical
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3, dtype="int32")
    model.add(tf.keras.layers.Dense(10, input_shape=(None, 4), activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, epochs=300, batch_size=10)
    return model.get_weights()


def train_keras_model(X_train, y_train, X_test, y_test):
    # Train using keras api and put weights in NN
    w = weigths_from_keras(X_train, y_train)
    ws = [w[0], w[2], w[4]]
    bs = [w[1], w[3], w[5]]
    nn = NN()
    predictions = tf.argmax(nn(X_test, ws, bs), 1).numpy()
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")


class NN(tf.Module):
    def __init__(self, name=None):
        super(NN, self).__init__(name=name)

    def __call__(self, x, w, b):
        x = tf.nn.relu(tf.tensordot(x, w[0], axes=1) + b[0])
        x = tf.nn.relu(tf.tensordot(x, w[1], axes=1) + b[1])
        return tf.nn.softmax(tf.tensordot(x, w[2], axes=1) + b[2])


def load_data():
    """
    Load and preprocess data
    """
    # Download data
    X, y = load_iris(return_X_y=True)
    # Split and convert to tensors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    X_train = tf.cast(X_train, dtype=tf.float32)
    y_train = tf.cast(y_train, dtype=tf.int32)
    X_test = tf.cast(X_test, dtype=tf.float32)
    y_test = tf.cast(y_test, dtype=tf.int32)
    return X_train, X_test, y_train, y_test


def initialize_population(population_size, weights_shapes, bias_shapes):
    """
    Create two lists with weights and biases
    """
    ws, bs = [], []
    for _ in range(0, population_size):
        # ws.append([tf.zeros(s, dtype=tf.float32) for s in weights_shapes])
        ws.append(
            [tf.random.uniform(s, -0.5, 0.5, dtype=tf.float32) for s in weights_shapes]
        )
        bs.append([tf.zeros(s, dtype=tf.float32) for s in bias_shapes])
    return ws, bs


# @tf.function
def calc_fitness(ws, bs, nn, X_train, y_train):
    """
    Calculate fitness for all individuals in the population
    """
    scores = []
    for w, b in zip(ws, bs):
        pred = tf.cast(tf.argmax(nn(X_train, w, b), 1), dtype=tf.int32)
        score = tf.reduce_mean(tf.cast(pred == y_train, dtype=tf.float32)).numpy()
        scores.append(score)
    return tf.cast(scores, dtype=tf.float32)


# @tf.function
def get_parent_indices(population_fitness):
    """
    Basically np.random.coice using probabilities. Reshape this to a tensor
    with shape (len(population_fitness), 2).
    """
    # from: https://stackoverflow.com/questions/41123879/numpy-random-choice-in-tensorflow
    cum_dist = tf.cast(tf.math.cumsum(population_fitness), dtype=tf.float32)
    cum_dist /= cum_dist[-1]
    unif_samp = tf.random.uniform((population_fitness.shape[0] - 1,), 0, 1)
    fitness = tf.reshape(
        tf.searchsorted(cum_dist, unif_samp), (population_fitness.shape[0] - 1, 1)
    )
    # Find index of fittest individual
    fittest_index = tf.reshape(
        tf.cast(tf.argmax(population_fitness), dtype=tf.int32), (1, 1)
    )
    return tf.concat([fittest_index, fitness], axis=0)


def mutate(tensor):
    targets = tf.random.uniform(tensor.shape, 0, 1, dtype=tf.float32)
    mutations = tf.random.normal(tensor.shape, -mut_range, mut_range, dtype=tf.float32)
    new_vals = tf.add(tensor, mutations)
    return tf.where(targets > mut_prob, tensor, new_vals)


def mutate_population(ws, bs, parent_indices):
    new_ws = [ws[i] for i in parent_indices[:, 0]]
    new_bs = [bs[i] for i in parent_indices[:, 0]]

    for i in range(1, len(new_ws)):
        new_ws[i][0] = mutate(new_ws[i][0])
        new_ws[i][1] = mutate(new_ws[i][1])
        new_ws[i][2] = mutate(new_ws[i][2])
        new_bs[i][0] = mutate(new_bs[i][0])
        new_bs[i][1] = mutate(new_bs[i][1])
        new_bs[i][2] = mutate(new_bs[i][2])

    return new_ws, new_bs


if __name__ == "__main__":
    # Set shapes of the weights and biases needed to run the NN
    # on the iris dataset
    weights_shapes = [(4, 10), (10, 10), (10, 3)]
    bias_shapes = [(10,), (10,), (3,)]
    population_size = 100
    n_generations = 100

    X_train, X_test, y_train, y_test = load_data()
    nn = NN()
    ws, bs = initialize_population(population_size, weights_shapes, bias_shapes)
    fitness = calc_fitness(ws, bs, nn, X_train, y_train)

    for gen in range(1, n_generations + 1):
        """
        # TST
        best_ind = tf.argmax(fitness).numpy()
        w_best = ws[best_ind]
        b_best = bs[best_ind]
        predictions = tf.argmax(nn(X_test, w_best, b_best), 1).numpy()
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
        """

        parent_indices = get_parent_indices(fitness)
        ws, bs = mutate_population(ws, bs, parent_indices)
        fitness = calc_fitness(ws, bs, nn, X_train, y_train)
        if gen % 10 == 0:
            print(f"{gen:5d}:, Max Fitness: {tf.reduce_max(fitness).numpy():.3f}")

    best_ind = tf.argmax(fitness)
    w_best = ws[parent_indices[best_ind][0]]
    b_best = bs[parent_indices[best_ind][0]]
    predictions = tf.argmax(nn(X_test, w_best, b_best), 1).numpy()
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
