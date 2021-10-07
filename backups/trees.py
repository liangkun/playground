#!/usr/bin/env python3

from math import log2
from operator import itemgetter
import pickle


def calc_shannon_entropy(data_set):
    """Calculate the Shannon entropy for data_set.

    data_set should be iterable. Each element of the data_set should be a sequence
    of features, with the last elements as class labels.
    """

    num_elements = len(data_set)
    labels = {}
    for data in data_set:
        label = data[-1]
        labels[label] = labels.get(label, 0.0) + 1

    shannon_entropy = 0.0
    for label_count in labels.values():
        label_prob = label_count / num_elements
        shannon_entropy -= label_prob * log2(label_prob)

    return shannon_entropy


def create_data_set():
    """Create a simple data set for testing.
    """
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    result = []
    for data in data_set:
        if data[axis] == value:
            new_data = data[:axis] + data[axis+1:]
            result.append(new_data)
    return result


def choose_best_feature(data_set):
    """Choose the best feature to split
    """

    least_entropy = calc_shannon_entropy(data_set)
    best_feature_idx = -1
    feature_count = len(data_set[0]) - 1
    data_count = len(data_set)
    for idx in range(feature_count):
        fvalue_set = {data[idx] for data in data_set}
        entropy = 0.0
        for value in fvalue_set:
            data_sub_set = split_data_set(data_set, idx, value)
            entropy += len(data_sub_set) / data_count * calc_shannon_entropy(data_sub_set)
        if entropy < least_entropy:
            least_entropy = entropy
            best_feature_idx = idx

    return best_feature_idx


def majority(label_list):
    label_count = {}
    for label in label_list:
        label_count[label] = label_count.get(label, 0) + 1
    sorted_label = sorted(label_count.items(), key=itemgetter(1), reverse=True)
    return sorted_label[0][0]


def create_tree(data_set, labels):
    class_list = [sample[-1] for sample in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    if len(data_set[0]) == 1:
        return majority(class_list)

    best_feature = choose_best_feature(data_set)
    best_feature_label = labels[best_feature]
    result_tree = {best_feature_label: {}}
    del labels[best_feature]
    fvalues = {sample[best_feature] for sample in data_set}
    for fvalue in fvalues:
        result_tree[best_feature_label][fvalue] = create_tree(
            split_data_set(data_set, best_feature, fvalue),
            labels[:])

    return result_tree


def classify(dtree, fvalues, flabels):
    if type(dtree) == str:
        return dtree
    else:
        root_label = list(dtree.keys())[0]
        fidx = flabels.index(root_label)
        fval = fvalues[fidx]
        return classify(dtree[root_label][fval], fvalues, flabels)


def store_tree(tree, fname):
    with open(fname, 'wb') as f:
        pickle.dump(tree, f)

def load_tree(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

