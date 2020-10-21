import datetime
from pathlib import Path
import string

import tensorflow as tf


def translate_string(s, genes):
    """
    Convert a strint to a list of indices based on genes content
    """
    l = []
    for c in s:
        i = genes.find(c)
        if i != -1:
            l.append(i)
        else:
            raise Exception
    return l


def translate_tensor(t, genes):
    """
    Convert a tensor to a string based on genes
    """
    s = ""
    for i in t.numpy():
        s += genes[i]
    return s


@tf.function
def create_initial_population(dna_length, population_size, genes):
    """
    Randomly create a tensor with the shape of (population_size, dna_length)
    with int32's between 0 and len(genes)
    """
    return tf.random.uniform(
        (population_size, dna_length),
        minval=0,
        maxval=len(genes),
        dtype=tf.int32,
    )


@tf.function
def create_target_population(population_size, target):
    """
    Convert the target into a tensor of size (population_size, len(target)
    which just repeats the target.
    """
    return tf.repeat([target], repeats=population_size, axis=0)


@tf.function
def calc_population_fitness(population, target_population):
    """
    Calculate the fitness of each row in the population. by counting
    how many numbers are equal to the target. Then scale it based on
    the length of a single target.
    """
    return (
        tf.math.count_nonzero(population == target_population, axis=1)
        / target_population.shape[1]
    )


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


@tf.function
def mate_parents(population, parent_indices):
    """
    Select parents from the population. Create a tensor with random
    values between 0 and 1 with the same shape as the population. Where
    values are above .5 choose dna from parent 1 else use dna from parent
    2.
    """
    p1 = tf.gather(population, parent_indices[:, 0])
    p2 = tf.gather(population, parent_indices[:, 1])
    probs = tf.random.uniform(population.shape, 0, 1, dtype=tf.float32)
    return tf.where(probs >= 0.5, p1, p2)


@tf.function
def mutate_population(population, mutation_rate, genes):
    """
    Create a probability tensorf with shape of population. Then create
    a random population and place dna from that one in the input
    population when prob in that position is below mutation rate.
    """
    probs = tf.random.uniform(population.shape, 0, 1, dtype=tf.float32)
    mutations = tf.random.uniform(population.shape, 0, len(genes), dtype=tf.int32)
    return tf.where(probs <= mutation_rate, mutations, population)


@tf.function
def next_generation(population, target_population, genes):
    """
    Calculate the population fitness, based on that, choose parents and create
    a new population form that.
    """
    population_fitness = calc_population_fitness(population, target_population)
    parent_indices = get_parent_indices(population_fitness)
    population = mutate_population(
        mate_parents(population, parent_indices), mutation_rate, genes
    )
    return population, population_fitness


def evolve(population, target, n_gen, genes, mutation_rate):
    """
    Run the genetic algorithm for n_gen times.
    """
    target_population = tf.constant(
        create_target_population(population_size, target), tf.int32
    )
    for g in tf.range(1, n_gen + 1):
        population, population_fitness = next_generation(
            population, target_population, genes
        )
        if g % 100 == 0:
            max_fitness = tf.reduce_max(population_fitness)
            tf.print(f"generation: {g}, Fitness: {max_fitness:.2f}")
    return population, population_fitness


if __name__ == "__main__":
    # Config
    population_size = 500
    mutation_rate = 0.001
    n_gen = 10000

    # Genes, this are all printable characters in a list
    genes = string.printable
    # Target to evolve to
    target_string = "The target of this algorithm."

    # Create a target tensor of int8s
    target = tf.constant(translate_string(target_string, genes), dtype=tf.int32)
    # Create a population with random tensors
    population = create_initial_population(target.shape[0], population_size, genes)
    # Evolve the population
    population, population_fitness = evolve(
        population, target, n_gen, genes, mutation_rate
    )

    # Sort population based on fitness
    inds = tf.argsort(population_fitness, -1)
    sorted_population = tf.gather(population, inds, batch_dims=0)

    # Show results
    print("-- TOP 10 INDIVIDUALS:")
    for i in population[:10]:
        print(translate_tensor(i, genes))
