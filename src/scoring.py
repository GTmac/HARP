#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""

import copy
import numpy
import pickle

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer

from collections import defaultdict
from gensim.models import Word2Vec

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels

def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in G.items()}

def scoring(embeddings, graph_file, training_percents, network_name="network", labels_name="group", number_shuffles=10, columnFilter=None):
    mat = loadmat(graph_file)
    A = mat[network_name]
    graph = sparse2graph(A)
    # 0. Load embeddings
    features_matrix = numpy.load(embeddings)

    # 1. Load labels
    labels_matrix = mat[labels_name]
    if type(labels_matrix) is numpy.ndarray:
        labels_matrix = csc_matrix(labels_matrix, dtype=numpy.int64)
    labels_count = labels_matrix.shape[1]
    mlb = MultiLabelBinarizer(range(labels_count))

    # 2. Shuffle, to create train/test groups
    shuffles = []
    for x in range(number_shuffles):
      shuffles.append(skshuffle(features_matrix, labels_matrix))

    # 3. to score each train/test group
    all_results = defaultdict(list)

    for train_percent in training_percents:
        print ('Training Percent: %.2f' % train_percent)
        for shuf in shuffles:

            X, y = shuf

            training_size = int(train_percent * X.shape[0])

            X_train = X[:training_size, :]
            y_train_ = y[:training_size]

            y_train = [[] for x in range(y_train_.shape[0])]

            cy =  y_train_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_train[i].append(j)

            assert sum(len(l) for l in y_train) == y_train_.nnz

            X_test = X[training_size:, :]
            y_test_ = y[training_size:]

            y_test = [[] for x in range(y_test_.shape[0])]

            cy =  y_test_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_test[i].append(j)

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train_)

            # find out how many labels should be predicted
            top_k_list = [len(l) for l in y_test]
            preds = clf.predict(X_test, top_k_list)

            results = {}
            averages = ["micro", "macro", "samples", "weighted"]
            for average in averages:
                results[average] = f1_score(mlb.fit_transform(y_test),  mlb.fit_transform(preds), average=average)

            all_results[train_percent].append(results)

    micro_f1 = {key: numpy.average([item['micro'] for item in all_results[key]]) for key in all_results}
    macro_f1 = {key: numpy.average([item['macro'] for item in all_results[key]]) for key in all_results}

    return all_results, micro_f1, macro_f1

if __name__ == '__main__':
    parser = ArgumentParser("embedding_scoring", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument("-e", "--embeddings", dest="embeddings", required=True, help="Embeddings file in NumPy .npy format.")
    parser.add_argument("-i", "--input", required=True,
            help="Input graph file. Must be a .mat file with graph adjacency matrix and node labels.")
    parser.add_argument("--adj-matrix-name", default='network',
            help="Variable name of adjacency matrix inside the .mat file.")
    parser.add_argument("--label-name", default='group',
            help="Variable name of node labels inside the .mat file.")
    parser.add_argument("-t", "--training_percents", type=int, required=True, nargs='+', help="Training set percentage.")
    args = parser.parse_args()
    args.training_percents = numpy.asarray(args.training_percents) * 0.01
    all_results, micro_f1, macro_f1 = scoring(args.embeddings, args.input, args.training_percents,
            network_name=args.adj_matrix_name, labels_name=args.label_name)
    for item in sorted(macro_f1):
        print ('Training ratio is %.1f%%, macro F1 is %.2f%%' % (item * 100, macro_f1[item] * 100))
