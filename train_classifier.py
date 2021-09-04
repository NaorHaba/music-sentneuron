import os
import csv
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


import midi_encoder as me
import plot_results as pr

from train_generative import build_generative_model
from sklearn.linear_model import LogisticRegression, LinearRegression
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Directory where trained model will be saved
TRAIN_DIR = "./trained"

class_dict = {0: {'emotion': 'happy', 'angle': 15},
              1: {'emotion': 'delighted', 'angle': 45},
              2: {'emotion': 'excited', 'angle': 75},
              3: {'emotion': 'tense', 'angle': 105},
              4: {'emotion': 'angry', 'angle': 135},
              5: {'emotion': 'frustrated', 'angle': 165},
              6: {'emotion': 'sad', 'angle': -165},
              7: {'emotion': 'depressed', 'angle': -135},
              8: {'emotion': 'tired', 'angle': -105},
              9: {'emotion': 'calm', 'angle': -75},
              10: {'emotion': 'relaxed', 'angle': -45},
              11: {'emotion': 'content', 'angle': -15}
              }

def preprocess_sentence(text, front_pad='\n ', end_pad=''):
    text = text.replace('\n', ' ').strip()
    text = front_pad+text+end_pad
    return text

def encode_sentence(model, text, char2idx, layer_idxs):
    text = preprocess_sentence(text)

    # Reset LSTMs hidden and cell states
    model.reset_states()

    for c in text.split(" "):
        # Add the batch dimension
        try:
            input_eval = tf.expand_dims([char2idx[c]], 0)
            predictions = model(input_eval)
        except KeyError:
            if c != "":
                print("Can't process char", c)
    h_states = []
    c_states = []
    for layer_idx in layer_idxs:
        h_state, c_state = model.get_layer(index=layer_idx).states
        h_states.append(tf.squeeze(h_state, 0))
        c_states.append(tf.squeeze(c_state, 0))
    c_state = np.concatenate(c_states)
    h_state = np.concatenate(h_states)
    # remove the batch dimension
    #h_state = tf.squeeze(h_state, 0)

    return tf.math.tanh(c_state).numpy()

# def build_dataset_classifier(datapath, generative_model, char2idx, layer_idx):
#     xs, ys = [], []
#
#     csv_file = open(datapath, "r")
#     data = csv.DictReader(csv_file)
#
#     for row in data:
#         arousal = row['arousal']
#         valence = row['valance']
#         label = int(row["label"])
#         filepath = row["midi"]
#
#         data_dir = os.path.dirname(datapath)
#         phrase_path = os.path.join(data_dir, filepath) + ".mid"
#         encoded_path = os.path.join(data_dir, filepath) + ".npy"
#
#         # Load midi file as text
#         if os.path.isfile(encoded_path):
#             encoding = np.load(encoded_path)
#         else:
#             text, vocab = me.load(phrase_path, transpose_range=1, stretching_range=1)
#
#             # Encode midi text using generative lstm
#             encoding = encode_sentence(generative_model, text, char2idx, layer_idx)
#
#             # Save encoding in file to make it faster to load next time
#             np.save(encoded_path, encoding)
#
#         xs.append(encoding)
#         ys.append(label)
#
#     return np.array(xs), np.array(ys)


def build_dataset(datapath, generative_model, char2idx, layer_idx, model_output):
    def get_label_classifier(valences, arousals, model_output):
        def radian_to_point(radian):
            return np.array([np.cos(radian), np.sin(radian)])

        results = []
        for v, a in zip(valences, arousals):
            results.append(min(class_dict.keys(),
                               key=lambda cl: np.linalg.norm(np.array([v, a]) -
                                                             radian_to_point(np.radians(class_dict[cl]['angle'])))
                               if cl in range(0, 12, 12 // model_output) else np.inf))
        return np.array(results)

    def calc_y(arousals, valences, model_output):
        if model_output == float("inf"):
            # min max normalization
            # arousals = (arousals - arousals.min()) / (arousals.max() - arousals.min())
            # TODO maybe change this return
            return valences #arousals * valences
        else:
            return get_label_classifier(valences, arousals, model_output)

    xs, ys, arousals, valences = [], [], [], []

    csv_file = open(datapath, "r")
    data = csv.DictReader(csv_file)

    for row in data:
        arousal = row['arousal']
        valence = row['valence']
        filepath = row["midi"]

        data_dir = os.path.dirname(datapath)
        # TODO find a better solution than '..' maybe change vigimidi_sent.csv
        path = os.path.join(data_dir, '..', filepath).replace('.mid', '')
        phrase_path = path + ".mid"
        encoded_path = path + ".npy"

        # Load midi file as text
        if os.path.isfile(encoded_path):
            encoding = np.load(encoded_path)
        else:
            text, vocab = me.load(phrase_path, transpose_range=1, stretching_range=1)

            # Encode midi text using generative lstm
            encoding = encode_sentence(generative_model, text, char2idx, layer_idx)

            # Save encoding in file to make it faster to load next time
            np.save(encoded_path, encoding)

        xs.append(encoding)
        arousals.append(arousal)
        valences.append(valence)
    ys = calc_y(np.array(arousals).astype(np.float), np.array(valences).astype(np.float), model_output)
    return np.array(xs), np.array(ys)


def get_model_coef(sent_model, model_output):
    # TODO change this according to the model type
    if model_output == float("inf"):
        return sent_model.coef_
    else:
        return sent_model.coef_[0]


def create_model(model_output):

    if model_output == float("inf"):
        return LinearRegression()
    else:
        return LogisticRegression()


def create_param_grid(model_output):
    if model_output == float("inf"):
        return {}
    else:
        return {'C': 2**np.arange(-8, 1).astype(np.float),
                'solver': ["liblinear"],
                'penalty': ['l1'],
                'random_state': [42]}


def get_score_metric(model_output):
    if model_output == float("inf"):
        return 'neg_mean_absolute_error'
    else:
        return 'f1_macro'



def train_model(train_dataset, test_dataset, model_output):
    model = create_model(model_output)
    param_grid = create_param_grid(model_output)
    score_metric = get_score_metric(model_output)

    trX, trY = train_dataset
    teX, teY = test_dataset
    search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, scoring=score_metric)
    search.fit(trX, trY)
    cv_score = search.best_score_
    print("cv_score", cv_score)
    # TODO decide if sent_pp needs to be a pipeline or only the model state in the pipeline
    sent_model = search.best_estimator_
    coef = get_model_coef(sent_model, model_output)

    score = sent_model.score(teX, teY)
    # Persist sentiment classifier
    with open(os.path.join(TRAIN_DIR, "classifier_ckpt.p"), "wb") as f:
        pickle.dump(sent_model, f)

    # Get activated neurons
    sentneuron_ixs = get_activated_neurons(sent_model, coef)

    # Plot results
    pr.plot_weight_contribs(coef)
    pr.plot_logits(trX, trY, sentneuron_ixs)

    return sentneuron_ixs, score



# def train_classifier_model(train_dataset, test_dataset, C=2**np.arange(-8, 1).astype(np.float), seed=42, penalty="l1"):
#
#     trX, trY = train_dataset
#     teX, teY = test_dataset
#
#     scores = []
#
#     # Hyper-parameter optimization
#     for i, c in enumerate(C):
#         logreg_model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i, solver="liblinear")
#         logreg_model.fit(trX, trY)
#
#         score = logreg_model.score(teX, teY)
#         scores.append(score)
#
#     c = C[np.argmax(scores)]
#
#     sent_classfier = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C), solver="liblinear")
#     sent_classfier.fit(trX, trY)
#
#     score = sent_classfier.score(teX, teY) * 100.
#
#     # Persist sentiment classifier
#     with open(os.path.join(TRAIN_DIR, "classifier_ckpt.p"), "wb") as f:
#         pickle.dump(sent_classfier, f)
#
#     # Get activated neurons
#     sentneuron_ixs = get_activated_neurons(sent_classfier)
#
#     # Plot results
#     pr.plot_weight_contribs(sent_classfier.coef_)
#     pr.plot_logits(trX, trY, sentneuron_ixs)
#
#     return sentneuron_ixs, score



# def train_classifier_model_regression(train_dataset, test_dataset, C=2**np.arange(-8, 1).astype(np.float), seed=42,
#                                       penalty="l1"):
#     trX, trY = train_dataset
#     teX, teY = test_dataset
#
#     # scores = []
#
#     # Hyper-parameter optimization
#     # for i, c in enumerate(C):
#     #     logreg_model = LinearRegression()
#     #     logreg_model.fit(trX, trY)
#     #
#     #     score = logreg_model.score(teX, teY)
#     #     scores.append(score)
#
#     # c = C[np.argmax(scores)]
#
#     sent_classfier = LinearRegression()
#     sent_classfier.fit(trX, trY)
#
#     score = sent_classfier.score(teX, teY)
#     print(score)
#
#     # Persist sentiment classifier
#     with open(os.path.join(TRAIN_DIR, "classifier_ckpt.p"), "wb") as f:
#         pickle.dump(sent_classfier, f)
#
#     # Get activated neurons
#     sentneuron_ixs = get_activated_neurons(sent_classfier)
#
#     # Plot results
#     pr.plot_weight_contribs(sent_classfier.coef_)
#     pr.plot_logits(trX, trY, sentneuron_ixs)
#
#     return sentneuron_ixs, score


def get_activated_neurons(sent_classfier, coef=None):
    if coef is None:
        coef = get_model_coef(sent_classfier)
    neurons_not_zero = len(np.argwhere(coef))

    weights = coef.T
    weights = weights.reshape(len(weights), 1)
    weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))
    # weight_penalties = abs(weights).sum()
    if neurons_not_zero == 1:
        neuron_ixs = np.array([np.argmax(weight_penalties)])
    elif neurons_not_zero >= np.log(len(weight_penalties)):
        neuron_ixs = np.argsort(weight_penalties)[-neurons_not_zero:][::-1]
    else:
        neuron_ixs = np.argpartition(weight_penalties, -neurons_not_zero)[-neurons_not_zero:]
        neuron_ixs = (neuron_ixs[np.argsort(weight_penalties[neuron_ixs])])[::-1]

    return neuron_ixs


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='train_classifier.py')
    parser.add_argument('--dataset', type=str, required=True, help="labeled dataset.")
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    parser.add_argument('--cellix', nargs='+', type=int, required=True, help="LSTM layer to use as encoder.")
    parser.add_argument('--model_output', type=int, default=2, help="amount of classes, infinity means regression")
    opt = parser.parse_args()


    # Load char2idx dict from json file
    with open(opt.ch2ix) as f:
        char2idx = json.load(f)

    # Calculate vocab_size from char2idx dict
    vocab_size = len(char2idx)

    # Rebuild generative model from checkpoint
    generative_model = build_generative_model(vocab_size, opt.embed, opt.units, opt.layers, batch_size=1)
    generative_model.load_weights(tf.train.latest_checkpoint(opt.model))
    generative_model.build(tf.TensorShape([1, None]))

    # Build dataset from encoded labelled midis
    # train_dataset = build_dataset(opt.train, generative_model, char2idx, opt.cellix, opt.model_output)
    x, y = build_dataset(opt.dataset, generative_model, char2idx, opt.cellix, opt.model_output)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    train_dataset, test_dataset = (x_train, y_train), (x_test, y_test)

    # Train model
    sentneuron_ixs, score = train_model(train_dataset, test_dataset, opt.model_output)

    print("Total Neurons Used:", len(sentneuron_ixs), "\n", sentneuron_ixs)
    print("Test Score:", score)
