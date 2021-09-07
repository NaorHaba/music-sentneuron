import os
import csv
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import midi_encoder as me
import plot_results as pr

from train_generative import build_generative_model
from sklearn.linear_model import LogisticRegression

# Directory where trained model will be saved
TRAIN_DIR = "./trained"

def preprocess_sentence(text, front_pad='\n ', end_pad=''):
    text = text.replace('\n', ' ').strip()
    text = front_pad+text+end_pad
    return text

def encode_sentence(model, text, char2idx, layer_idx):
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

    h_state, c_state = model.get_layer(index=layer_idx).states

    # remove the batch dimension
    #h_state = tf.squeeze(h_state, 0)
    c_state = tf.squeeze(c_state, 0)

    return tf.math.tanh(c_state).numpy()

def build_dataset(datapath, generative_model, char2idx, layer_idx):
    xs, ys = [], []

    csv_file = open(datapath, "r")
    data = csv.DictReader(csv_file)

    for row in data:
        label = int(row["label"])
        filepath = row["filepath"]

        data_dir = os.path.dirname(datapath)
        phrase_path = os.path.join(data_dir, filepath) + ".mid"
        encoded_path = os.path.join(data_dir, filepath) + ".npy"

        # Load midi file as text
        # if os.path.isfile(encoded_path):
        #     encoding = np.load(encoded_path)
        # else:
        text, vocab = me.load(phrase_path, transpose_range=1, stretching_range=1)

        # Encode midi text using generative lstm
        encoding = encode_sentence(generative_model, text, char2idx, layer_idx)

        # Save encoding in file to make it faster to load next time
        np.save(encoded_path, encoding)

        xs.append(encoding)
        ys.append(label)

    return np.array(xs), np.array(ys)

def train_classifier_model(train_dataset, test_dataset, C=2**np.arange(-8, 1).astype(np.float), seed=42, penalty="l1"):
    trX, trY = train_dataset
    teX, teY = test_dataset

    scores = []


    # Hyper-parameter optimization
    for i, c in enumerate(C):
        logreg_model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i, solver="liblinear")
        logreg_model.fit(trX, trY)

        score = logreg_model.score(teX, teY)
        scores.append(score)

    c = C[np.argmax(scores)]

    sent_classfier = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C), solver="liblinear")
    sent_classfier.fit(trX, trY)

    score =  sent_classfier.score(teX, teY) * 100.

    # Persist sentiment classifier
    with open(os.path.join(TRAIN_DIR, "classifier_ckpt.p"), "wb") as f:
        pickle.dump(sent_classfier, f)

    # Get activated neurons
    sentneuron_ixs = get_activated_neurons(sent_classfier)

    # Plot results
    pr.plot_weight_contribs(sent_classfier.coef_)
    pr.plot_logits(trX, trY, sentneuron_ixs)

    return sentneuron_ixs, score

def get_activated_neurons(sent_classfier):
    neurons_not_zero = len(np.argwhere(sent_classfier.coef_))

    weights = sent_classfier.coef_.T
    weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))

    if neurons_not_zero == 1:
        neuron_ixs = np.array([np.argmax(weight_penalties)])
    elif neurons_not_zero >= np.log(len(weight_penalties)):
        neuron_ixs = np.argsort(weight_penalties)[-neurons_not_zero:][::-1]
    else:
        neuron_ixs = np.argpartition(weight_penalties, -neurons_not_zero)[-neurons_not_zero:]
        neuron_ixs = (neuron_ixs[np.argsort(weight_penalties[neuron_ixs])])[::-1]

    return neuron_ixs


def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')
    feat_imp = pd.Series(results.best_estimator_.feature_importances_).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


def train_classifier_model_new(train_dataset, test_dataset, C=2 ** np.arange(-8, 1).astype(np.float), seed=42,
                               penalty="l1"):
    trX, trY = train_dataset
    teX, teY = test_dataset

    # trainX, trainY, valX, valy = train_test_split(trX, trY, test_size=0.2, random_state=42)
    # 200, 2, 51, 382, 16, 0.1, 0.8
    # param_test1 = {'n_estimators': range(20, 500, 10)}
    # gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,
    #                                                              min_samples_leaf=50, max_depth=8, max_features='sqrt',
    #                                                              subsample=0.8, random_state=10),
    #                         param_grid=param_test1, scoring='roc_auc', n_jobs=-1, cv=5)
    # gsearch1.fit(trX, trY)
    # display(gsearch1)
    #
    # param_test2 = {'max_depth': range(1, 16, 1), 'min_samples_split': range(2, 1001, 20), 'min_samples_leaf': range(1, 71, 5)}
    # gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],
    #                                                              max_features='sqrt', subsample=0.8, random_state=10),
    #                         param_grid=param_test2, scoring='roc_auc', n_jobs=-1, cv=5)
    # gsearch2.fit(trX, trY)
    #
    # display(gsearch2)
    #
    # param_test4 = {'max_features': range(2, 70, 2)}
    # gsearch4 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1,
    #                                                              n_estimators=gsearch1.best_params_['n_estimators'],
    #                                                              max_depth=gsearch2.best_params_['max_depth'],
    #                                                              max_features='sqrt', subsample=0.8, random_state=10,
    #                                                              min_samples_split=gsearch2.best_params_['min_samples_split'],
    #                                                              min_samples_leaf=gsearch2.best_params_['min_samples_leaf']),
    #                         param_grid=param_test4, scoring='roc_auc', n_jobs=-1, cv=5)
    # gsearch4.fit(trX, trY)
    #
    # display(gsearch4)
    #
    # param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0], 'learning_rate': [0.1, 0.05, 0.01]}
    # gsearch5 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1,
    #                                                              n_estimators=gsearch1.best_params_['n_estimators'],
    #                                                              max_depth=gsearch2.best_params_['max_depth'],
    #                                                              max_features=gsearch4.best_params_['max_features'],
    #                                                              subsample=0.8, random_state=10,
    #                                                              min_samples_split=gsearch2.best_params_['min_samples_split'],
    #                                                              min_samples_leaf=gsearch2.best_params_['min_samples_leaf']),
    #                         param_grid=param_test5, scoring='roc_auc', n_jobs=-1, cv=5)
    # gsearch5.fit(trX, trY)
    #
    # display(gsearch5)
    # Hyper-parameter optimization
    # for i, c in enumerate(C):
    #     logreg_model = LogisticRegression(C=c, penalty=penalty, random_state=seed + i, solver="liblinear")
    #     logreg_model.fit(trX, trY)
    #
    #     score = logreg_model.score(teX, teY)
    #     scores.append(score)
    #
    # c = C[np.argmax(scores)]
    #
    # sent_classfier = LogisticRegression(C=c, penalty=penalty, random_state=seed + len(C), solver="liblinear")

    sfs = SequentialFeatureSelector(LogisticRegression(penalty="l1", random_state=42, solver="liblinear"),
                                    n_features_to_select=50, n_jobs=-1)
    trX = sfs.fit_transform(trX, trY)
    teX = sfs.transform(teX)
    print(sfs.get_support())
    lr = LogisticRegression(random_state=42, solver="liblinear", max_iter=1000)
    param_grid = {'C': 2**np.arange(-8, 1).astype(np.float),
                  'penalty': ["l1", "l2", 'elasticnet'],
                  'solver': ['lbfgs', 'liblinear']}
    grid = GridSearchCV(lr, param_grid, refit=True, verbose=2, n_jobs=-1)
    grid.fit(trX, trY)
    print(grid.best_estimator_)
    # sent_classfier.fit(trX, trY)

    score = grid.score(teX, teY) * 100.
    print(score)

    # Persist sentiment classifier
    with open(os.path.join(TRAIN_DIR, "classifier_ckpt.p"), "wb") as f:
        pickle.dump(sent_classfier, f)

    # Get activated neurons
    sentneuron_ixs = get_activated_neurons(sent_classfier)

    # Plot results
    pr.plot_weight_contribs(sent_classfier.coef_)
    pr.plot_logits(trX, trY, sentneuron_ixs)

    return sentneuron_ixs, score


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='train_classifier.py')
    parser.add_argument('--train', type=str, required=True, help="Train dataset.")
    parser.add_argument('--test' , type=str, required=True, help="Test dataset.")
    parser.add_argument('--model', type=str, required=True, help="Checkpoint dir.")
    parser.add_argument('--ch2ix', type=str, required=True, help="JSON file with char2idx encoding.")
    parser.add_argument('--embed', type=int, required=True, help="Embedding size.")
    parser.add_argument('--units', type=int, required=True, help="LSTM units.")
    parser.add_argument('--layers', type=int, required=True, help="LSTM layers.")
    parser.add_argument('--cellix', type=int, required=True, help="LSTM layer to use as encoder.")
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
    train_dataset = build_dataset(opt.train, generative_model, char2idx, opt.cellix)
    test_dataset = build_dataset(opt.test, generative_model, char2idx, opt.cellix)

    # Train model
    sentneuron_ixs, score = train_classifier_model(train_dataset, test_dataset)

    print("Total Neurons Used:", len(sentneuron_ixs), "\n", sentneuron_ixs)
    print("Test Accuracy:", score)
