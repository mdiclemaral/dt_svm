"""
Maral Dicle Maral June 2021

In the first part, Decision tree with information gain and gain ratio models was implemented.
Binary tree is used. Threshold values were determined according to best entropy.

In the second part, SVM algorithm on a breast cancer dataset was used.
LIBSVM library was used. Min-max normalization is applied.

"""
import sys
import pandas as pd
import numpy as np
from libsvm.svmutil import *


######  PART 1  ###### #Decision Tree Implementation

class Node:
    def __init__(self, data, current_info, parent, depth):  # Node constructor
        self.data = data
        self.current_info = current_info
        self.parent = parent
        self.attribute = None
        self.children = None
        self.treshold = None
        self.depth = depth

    def split(self, gain_type):  # Splits the current data node into two according to the best information gain
        best_attribute, best_first, best_second, best_info, best_treshold, best_entropy = 999, [], [], 0, 0, 1
        for attribute in range(0, 4):
            for datum in self.data:
                first, second = self.split_test(attribute, self.data, datum[attribute])
                entropy = self.weighted_entropy(first, second)
                info = np.abs(self.current_info - entropy)
                if info == 1:
                    best_attribute = attribute
                    best_first = first
                    best_second = second
                    best_treshold = datum[attribute]
                    break
                if gain_type == "gain_ratio":
                    if len(first) == 0 or len(second) == 0:
                        continue
                    else:
                        ratio = self.gain_ratio(first, second)
                        info = (info / ratio)
                if info >= best_info:
                    best_attribute = attribute
                    best_first = first
                    best_second = second
                    best_info = info
                    best_treshold = datum[attribute]
            else:
                continue
            break
        self.treshold = best_treshold
        self.attribute = best_attribute
        left = Node(best_first, self.entropy(best_first), self, self.depth+1)
        right = Node(best_second, self.entropy(best_second), self, self.depth+1)
        self.children = [left, right]

        return self.children

    def split_test(self, attribute, data, threshold):  # Node data is splitted according to given threshold
        first, second = [], []
        for datum in data:
            if datum[attribute] <= threshold:
                first.append(datum)
            else:
                second.append(datum)
        return first, second

    def gain_ratio(self, left, right):  # Split ratio
        l = len(left) / len(self.data)
        r = len(right) / len(self.data)
        return -1 * (l * np.log(l) + r * np.log(r))

    def weighted_entropy(self, left, right):  # Total entropy of two data lists
        if len(left) == 0 or len(right) == 0:
            return 1
        e_l = self.entropy(left)
        e_r = self.entropy(right)
        return e_l + e_r

    def entropy(self, data):  # Entropy of a given set of data
        l0, l1 = self.counter(data)
        if l0 == 0 or l1 == 0:
            e_l = 0
        else:
            e_l = (((l0 / (l0 + l1)) * np.log(l0 / (l0 + l1))) + ((l1 / (l0 + l1)) * np.log(l1 / (l0 + l1))))
        return -1 * e_l

    def counter(self, data):  # Counts positive and negative y values of given data
        c0 = 0
        c1 = 0
        for l in data:
            if l[4] == 0:
                c0 += 1
            else:
                c1 += 1
        return c0, c1


def file_handler(path):  # Extracts data from the given csv file to list of lists
    test = []
    train = []

    df = pd.read_csv(path, sep=',', header=None)

    attributes = df.iloc[:1].values.tolist()[0]

    df = df.iloc[1:]
    df[4] = df[4].str.contains("Iris-setosa").astype(int) # Iris - setosa = class1  Iris-versicolor = class0
    df = df.astype(float)
    df1 = df[df[4].isin([1])]
    df0 = df[df[4].isin([0])]

    train.extend(df1.head(40).values.tolist())
    test.extend(df1.tail(10).values.tolist())

    train.extend(df0.head(40).values.tolist())
    test.extend(df0.tail(10).values.tolist())

    return train, test, attributes


def train_tree(train_node, max_depth, gain_type):  # Trains a decision tree
    queue = [train_node]
    depth = 0
    while True:
        visiting_node = queue.pop(0)
        if visiting_node.current_info == 0:
            return depth
        if visiting_node.depth >= max_depth:
            return max_depth
        children = visiting_node.split(gain_type)
        queue.extend(children)
        depth = children[0].depth


def predict(train_node, depth, test):  # Predicts y values of given data points according to the formed decision tree
    y_pred = []
    for t in test:
        queue = [train_node]
        for d in range(0, depth):
            visiting_node = queue.pop(0)
            if visiting_node.attribute == None: #Reached a leaf node, stop the search
                break
            if t[visiting_node.attribute] <= visiting_node.treshold:
                queue.append(visiting_node.children[0])
            else:
                queue.append(visiting_node.children[1])
        visiting_node = queue.pop(0)
        positives, negatives = visiting_node.counter(visiting_node.data)
        p = positives / (positives + negatives)
        if p >= 0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred

######   PART 2  ######  # SVM with libsvm


def file_handler_svm(path):  # Extracts data from the given csv file to list of lists for svm program

    df = pd.read_csv(path, sep=',', header=None)
    df = df.iloc[1:]
    y = df[1].str.contains('M').astype(int).tolist() # M class: 1      B class: -1
    df = df.drop([0,1,32], axis=1)
    df = df.astype(float)
    df = (df - df.min()) / (df.max() - df.min())
    x = df.values.tolist()

    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    return x, y

part = sys.argv[1]
step = sys.argv[2]

if part == "part1": # Decision Tree
    train, test, attributes = file_handler("iris.csv")
    train_node = Node(train, 1, None, 0)
    if step == "step1":
        depth = train_tree(train_node, 7, "ig")
    else:
        depth = train_tree(train_node, 7, "gain_ratio")
    y_pred = predict(train_node, depth, test)
    count = 0
    for t in range(0, len(test)):
        if test[t][4] == y_pred[t]:
            count += 1
    print("DT", attributes[train_node.attribute], count/len(test))

if part == "part2": # SVM
    X, Y = file_handler_svm("wbcd.csv")
    if step == "step1":
        a = svm_train(Y[:400], X[:400], '-c 0.1 -t 0 -q')
        a_label, a_acc, a_val = svm_predict(Y[400:], X[400:], a, "-q")
        svs_a = a.get_SV()
        print('SVM kernel=linear C=0.1 acc={} n={}'.format(a_acc[0] / 100,len(svs_a)))

        b = svm_train(Y[:400], X[:400], '-c 1 -t 0 -q')
        b_label, b_acc, b_val = svm_predict(Y[400:], X[400:], b, "-q")
        svs_b = b.get_SV()
        print('SVM kernel=linear C=1 acc={} n={}'.format(b_acc[0] / 100, len(svs_b)))

        c = svm_train(Y[:400], X[:400], '-c 10 -t 0 -q')
        c_label, c_acc, c_val = svm_predict(Y[400:], X[400:], c, "-q")
        svs_c = c.get_SV()
        print('SVM kernel=linear C=10 acc={} n={}'.format(c_acc[0] / 100, len(svs_c)))

        d = svm_train(Y[:400], X[:400], '-c 20 -t 0 -q')
        d_label, d_acc, d_val = svm_predict(Y[400:], X[400:], d, "-q")
        svs_d = d.get_SV()
        print('SVM kernel=linear C=20 acc={} n={}'.format(d_acc[0] / 100, len(svs_d)))

        e = svm_train(Y[:400], X[:400], '-c 50 -t 0 -q')
        e_label, e_acc, e_val = svm_predict(Y[400:], X[400:], e, "-q")
        svs_e = e.get_SV()
        print('SVM kernel=linear C=50 acc={} n={}'.format(e_acc[0] / 100, len(svs_e)))
    else:
        a = svm_train(Y[:400], X[:400], '-t 0 -q')
        a_label, a_acc, a_val = svm_predict(Y[400:], X[400:], a, "-q")
        svs_a = a.get_SV()
        print('SVM kernel=linear C=1 acc={} n={}'.format(a_acc[0] / 100, len(svs_a)))

        b = svm_train(Y[:400], X[:400], '-t 1 -q')
        b_label, b_acc, b_val = svm_predict(Y[400:], X[400:], b, "-q")
        svs_b = b.get_SV()
        print('SVM kernel=polynomial C=1 acc={} n={}'.format(b_acc[0] / 100, len(svs_b)))

        c = svm_train(Y[:400], X[:400], '-t 2 -q')
        c_label, c_acc, c_val = svm_predict(Y[400:], X[400:], c, "-q")
        svs_c = c.get_SV()
        print('SVM kernel=radialbasisfunction C=1 acc={} n={}'.format(c_acc[0] / 100, len(svs_c)))

        d = svm_train(Y[:400], X[:400], '-t 3 -q')
        d_label, d_acc, d_val = svm_predict(Y[400:], X[400:], d, "-q")
        svs_d = d.get_SV()
        print('SVM kernel=sigmoid C=1 acc={} n={}'.format(d_acc[0] / 100, len(svs_d)))



