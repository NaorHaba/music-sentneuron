import os
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf

from midi_encoder import parse_midi
from midi_generator import generate_midi
from train_classifier import encode_sentence, class_dict
from train_classifier import get_activated_neurons
from train_generative import build_generative_model

GEN_MIN = -1
GEN_MAX =  1

# Directory where trained model will be saved
TRAIN_DIR = "./trained"


def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.uniform(0, 1) < mutation_rate:
            individual[i] = np.random.uniform(GEN_MIN, GEN_MAX)


def crossover(parent_a, parent_b, ind_size):
    # Averaging crossover
    return (parent_a + parent_b)/2


def reproduce(mating_pool, new_population_size, ind_size, mutation_rate):
    new_population = np.zeros((new_population_size, ind_size))

    for i in range(new_population_size):
        a = np.random.randint(len(mating_pool))
        b = np.random.randint(len(mating_pool))

        new_population[i] = crossover(mating_pool[a], mating_pool[b], ind_size)

    # Mutate new children
    np.apply_along_axis(mutation, 1, new_population, mutation_rate)

    return new_population


def roulette_wheel(population, fitness_pop):
    # Normalize fitnesses
    norm_fitness_pop = fitness_pop/np.sum(fitness_pop)

    # Here all the fitnesses sum up to 1
    r = np.random.uniform(0, 1)

    fitness_so_far = 0
    for i in range(len(population)):
        fitness_so_far += norm_fitness_pop[i]

        if r < fitness_so_far:
            return population[i]

    return population[-1]


def select(population, fitness_pop, mating_pool_size, ind_size, elite_rate):
    mating_pool = np.zeros((mating_pool_size, ind_size))

    # Apply roulete wheel to select mating_pool_size individuals
    for i in range(mating_pool_size):
        mating_pool[i] = roulette_wheel(population, fitness_pop)

    # Apply elitism
    assert elite_rate >= 0 and elite_rate <= 1
    elite_size = int(np.ceil(elite_rate * len(population)))
    elite_idxs = np.argsort(-fitness_pop, axis=0)

    for i in range(elite_size):
        r = np.random.randint(0, mating_pool_size)
        mating_pool[r] = population[elite_idxs[i]]

    return mating_pool


def calc_fitness(individual, gen_model, cls_model, char2idx, idx2char, layer_idxs, sentiment, activated_neurons_file, runs=30):
    encoding_size = 0
    for layer_idx in layer_idxs:
        encoding_size += gen_model.get_layer(index=layer_idx).units
    generated_midis = np.zeros((runs, encoding_size))

    # Get activated neurons
    if opt.sent == 'regression':
        sentneuron_ixs = get_activated_neurons(activated_neurons_file)
    else:
        sentneuron_ixs = get_activated_neurons(activated_neurons_file)

    assert len(individual) == len(sentneuron_ixs), "assert in calc_fitness failed!, len of ind neq to len of sentneuron"

    # Use individual gens to override model neurons
    override = {}
    for i, ix in enumerate(sentneuron_ixs):
        override[ix] = individual[i]

    # Generate pieces and encode them using the cell state of the generative model
    for i in range(runs):
        midi_text = generate_midi(gen_model, char2idx, idx2char, seq_len=64, layer_idxs=layer_idxs, override=override)
        generated_midis[i] = encode_sentence(gen_model, midi_text, char2idx, layer_idxs)

    generated_midis = generated_midis[:, sentneuron_ixs]
    midis_sentiment = cls_model.predict(generated_midis).clip(min=0)

    if opt.sent == 'regression':
        return - np.sum(np.abs(midis_sentiment - sentiment)) / runs  # negative MAE
    else:
        return np.sum((midis_sentiment == sentiment)) / runs  # accuracy


def evaluate(population, gen_model, cls_model, char2idx, idx2char, layer_idx, sentiment, activated_neurons_file):
    fitness = np.zeros((len(population), 1))

    for i in range(len(population)):
        fitness[i] = calc_fitness(population[i], gen_model, cls_model, char2idx, idx2char, layer_idx, sentiment, activated_neurons_file)

    return fitness


def evolve(pop_size, ind_size, mut_rate, elite_rate, epochs, sentiment, activated_neurons_file):
    # Create initial population
    population = np.random.uniform(GEN_MIN, GEN_MAX, (pop_size, ind_size))

    # Evaluate initial population
    fitness_pop = evaluate(population, gen_model, cls_model, char2idx, idx2char, opt.cellix, sentiment, activated_neurons_file)
    print("--> Fitness: \n", fitness_pop)

    for i in range(epochs):
        print("-> Epoch", i)

        # Select individuals via roulette wheel to form a mating pool
        mating_pool = select(population, fitness_pop, pop_size, ind_size, elite_rate)

        # Reproduce mating pool with crossover and mutation to form new population
        population = reproduce(mating_pool, pop_size, ind_size, mut_rate)

        # Calculate fitness of each individual of the population
        fitness_pop = evaluate(population, gen_model, cls_model, char2idx, idx2char, opt.cellix, sentiment, activated_neurons_file)
        print("--> Fitness: \n", fitness_pop)

    return population, fitness_pop


def calc_regression_value(cls_model, example_midi, sentneuron_ixs):
    defaults = dict(sample_freq=4, piano_range=(33, 93), transpose_range=10, stretching_range=10)
    midi_text = parse_midi(example_midi, **defaults)
    encoded_midi = encode_sentence(gen_model, midi_text, char2idx, opt.cellix)
    predicted_sent = cls_model.predict(encoded_midi.reshape(1, -1)[:, sentneuron_ixs]).clip(min=0)[0]  # TODO maybe clip not required when predicting for 1 sample
    return predicted_sent


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='evolve_generative.py')
    parser.add_argument('--genmodel', type=str, required=True, help="Generative model to evolve.")
    parser.add_argument('--clsmodel', type=str, required=True, help="Classifier model to calculate fitness.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    parser.add_argument('--cellix', nargs='+', type=int, required=True, help="LSTM layer to use as encoder.")
    parser.add_argument('--sent', type=str, required=True,
                        help="Desired emotion from the class_dict defined in train_classifier or 'regression'.")
    parser.add_argument('--popsize', type=int, default=10, help="Population size.")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs to run.")
    parser.add_argument('--mrate', type=float, default=0.1, help="Mutation rate.")
    parser.add_argument('--elitism', type=float, default=0.0, help="Elitism in percentage.")
    parser.add_argument('--example_midi', type=str,
                        default=r"..\vgmidi\labelled\phrases\Banjo-Kazooie_N64_Banjo-Kazooie_Boggys Igloo Happy_0.mid",
                        help="path to example midi required for regression sentiment.")
    parser.add_argument('--activated_neurons_file', type=str,
                        default='trained/activated_neurons.npy',
                        help="path to file saving the activated neurons indices")

    opt = parser.parse_args()

    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Create idx2char from char2idx dict
    idx2char = {idx:char for char, idx in char2idx.items()}

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild generative model from checkpoint
    gen_model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    gen_model.load_weights(tf.train.latest_checkpoint(opt.genmodel))
    gen_model.build(tf.TensorShape([1, None]))

    # Load classifier model
    with open(opt.clsmodel, "rb") as f:
        cls_model = pickle.load(f)

    # encode the desired emotion or calculate it if regression
    if opt.sent == 'regression':
        # Set individual size to the number of activated neurons
        sentneuron_ixs = get_activated_neurons(opt.activated_neurons_file)
        sentiment = calc_regression_value(cls_model, opt.example_midi, sentneuron_ixs)
        print(f'The sentiment value of the provided example_midi file is {sentiment}')
    else:
        sentneuron_ixs = get_activated_neurons(opt.activated_neurons_file)
        try:
            sentiment = [cl for cl in class_dict if class_dict[cl]['emotion'] == opt.sent][0]
        except KeyError as exc:
            raise ValueError('The given sentiment is not present in the class_dict') from exc

    ind_size = len(sentneuron_ixs)


    population, fitness_pop = evolve(opt.popsize, ind_size, opt.mrate, opt.elitism, opt.epochs, sentiment, opt.activated_neurons_file)


    # Get best individual
    best_idx = np.argmax(fitness_pop)
    best_individual = population[best_idx]

    # Use best individual gens to create a dictionary with cell values
    neurons = {}
    for i, ix in enumerate(sentneuron_ixs):
        neurons[str(ix)] = best_individual[i]

    print(neurons)

    with open(os.path.join(TRAIN_DIR, "neurons_" + opt.sent + ".json"), "w") as f:
        json.dump(neurons, f)
