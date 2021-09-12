import os
import csv
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.svm import SVR, SVC

import midi_encoder as me

from train_generative import build_generative_model

# Directory where trained model will be saved
TRAIN_DIR = "./trained"

# dictionary encoding emotions and their place along the "unit circle" of emotions as seen in the article
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


def get_states_from_layers(model, layer_idxs):
    h_states = []
    c_states = []
    for layer_idx in layer_idxs:
        h_state, c_state = model.get_layer(index=layer_idx).states
        h_states.append(tf.squeeze(h_state, 0))
        c_states.append(tf.squeeze(c_state, 0))
    # return a concatenated vector including the states from the received layers
    c_state = np.concatenate(c_states)
    h_state = np.concatenate(h_states)
    return h_state, c_state


def encode_sentence(model, text, char2idx, layer_idxs):
    text = preprocess_sentence(text)

    # Reset LSTMs hidden and cell states
    model.reset_states()

    for c in text.split(" "):
        # Add the batch dimension
        try:
            input_eval = tf.expand_dims([char2idx[c]], 0)
            model(input_eval)
        except KeyError:
            if c != "":
                print("Can't process char", c)
    h_state, c_state = get_states_from_layers(model, layer_idxs)

    return tf.math.tanh(c_state).numpy()


def select_features(x_train, y_train, x_test, portion, model_output, activated_neurons_file):
    if model_output == float("inf"):
        estimator = SVR(kernel="linear")
    else:
        estimator = SVC(kernel="linear")
    rfe_selector = RFE(estimator, n_features_to_select=portion, step=1)
    selector = rfe_selector.fit(x_train, y_train)
    rfe_support = selector.support_

    # save activated neurons
    np.save(activated_neurons_file, np.where(rfe_support))
    return x_train[:, rfe_support], x_test[:, rfe_support]


def build_dataset(datapath, generative_model, char2idx, layer_idx, model_output):

    def get_classifier_label(valences, arousals, model_output):
        def radian_to_point(radian):
            return np.array([np.cos(radian), np.sin(radian)])

        results = []
        for v, a in zip(valences, arousals):
            # add the closest point (representing emotion) to the given (v, a) point
            results.append(min(class_dict.keys(),
                               key=lambda cl: np.linalg.norm(np.array([v, a]) -
                                                             radian_to_point(np.radians(class_dict[cl]['angle'])))
                               if cl in range(0, 12, 12 // model_output) else np.inf))
        return np.array(results)

    def calc_y(arousals, valences, model_output):
        if model_output == float("inf"):
            return valences
        else:
            return get_classifier_label(valences, arousals, model_output)

    xs, ys, arousals, valences = [], [], [], []

    csv_file = open(datapath, "r")
    data = csv.DictReader(csv_file)

    for row in data:
        arousal = row['arousal']
        valence = row['valence']
        filepath = row["midi"]

        data_dir = os.path.dirname(datapath)
        path = os.path.join(data_dir, filepath).replace('.mid', '')
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
    ys = np.array(calc_y(np.array(arousals).astype(float), np.array(valences).astype(float), model_output))
    xs = np.array(xs)
    return xs, ys


def create_model(model_output):
    if model_output == float("inf"):
        return SVR()
    else:
        return SVC()


def create_param_grid(model_output):
    if model_output == float("inf"):
        return {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    else:
        return {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}


def get_score_metric(model_output):
    if model_output == float("inf"):
        return 'neg_mean_absolute_error'
    else:
        return 'accuracy'


def train_model(train_dataset, test_dataset, model_output):
    model = create_model(model_output)
    param_grid = create_param_grid(model_output)
    score_metric = get_score_metric(model_output)

    trX, trY = train_dataset
    teX, teY = test_dataset
    # train a GridSearch to find the best parameters of the model
    search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, scoring=score_metric)
    search.fit(trX, trY)
    cv_score = search.best_score_
    print("cv_score", cv_score)
    sent_model = search.best_estimator_

    score = sent_model.score(teX, teY)
    # Persist sentiment classifier
    with open(os.path.join(TRAIN_DIR, "classifier_ckpt.p"), "wb") as f:
        pickle.dump(sent_model, f)

    return score


def get_activated_neurons(activated_neurons_file):
    activated_neurons = np.load(activated_neurons_file)
    return activated_neurons.reshape(activated_neurons.shape[1])


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
    parser.add_argument('--model_output', type=int, default=2,
                        help="amount of classes, infinity means regression")
    parser.add_argument('--n_features', type=float, default=0.1, help="amount of classes, infinity means regression")
    parser.add_argument('--activated_neurons_file', type=str,
                        default='trained/activated_neurons.npy', help="path to file saving the activated neurons indices")
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
    x, y = build_dataset(opt.dataset, generative_model, char2idx, opt.cellix, opt.model_output)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_test = select_features(x_train, y_train, x_test, opt.n_features, opt.model_output, opt.activated_neurons_file)
    train_dataset, test_dataset = (x_train, y_train), (x_test, y_test)

    # Train model
    score = train_model(train_dataset, test_dataset, opt.model_output)
    sentneuron_ixs = np.load(opt.activated_neurons_file)

    print("Total Neurons Used:", len(sentneuron_ixs[0]), "\n", sentneuron_ixs)
    print("Test Score:", score)


    # --model trained --ch2ix trained/char2idx.json --embed 256 --units 256 --layers 4 --dataset ..\vgmidi\vgmidi.csv --cellix 1 2 3 4