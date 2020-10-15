# Genetic Algorithm in Tensorflow

A simple implementation of a genetic algorithm in tensorflow.

## Speed Up due to GPU

Simple timer test on google colab:

* Target: `The target of this algorithm. This can be quite long`
* population_size = 1000
* mutation_rate = 0.001
* n_gen = 100000

## Results

| CPU       | GPU       |
| --------- | --------- |
| 258.59 s. | 186.52 s. |
