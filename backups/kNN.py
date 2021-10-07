#!/usr/bin/env python3

import numpy as np
from math import floor
from operator import itemgetter
from os import listdir


def create_data_set():
    samples = np.array([[0.0, 0.1], [0.0, 0.0], [1.1, 0.9], [1.0, 1.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    return samples, labels


def classify0(x, samples, labels, k):
    """Simple classifier using kNN."""

    # compute the distances
    sample_size = samples.shape[0]
    distances = np.sum((np.tile(x, (sample_size, 1)) - samples) ** 2, axis=1) ** 0.5
    inc_index = np.argsort(distances)

    # get he neighbour's votes
    votes = {}
    for i in range(k):
        vote_label = labels[inc_index[i]]
        votes[vote_label] = votes.get(vote_label, 0) + 1
    sorted_votes = sorted(votes.items(), key=itemgetter(1), reverse=True)

    return sorted_votes[0][0]


def file2matrix(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        num = len(lines)
        samples = np.zeros((num, 3))
        labels = np.array(['<unknown>'] * num)
        for idx, line in enumerate(lines):
            elements = line.strip().split('\t')
            samples[idx, :] = elements[:-1]
            labels[idx] = elements[-1]

        return samples, labels


def normalize(data):
    """Normalize the values in date into range [0, 1].

     new_value = (old_value - min) / (max - min)
    """
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    ranges = max_values - min_values
    return (data - min_values) / ranges, ranges, min_values


def dating_classify_test(filename='data/Ch02/datingTestSet.txt', k=3, ratio=0.1):
    samples, labels = file2matrix(filename)
    samples, _, _ = normalize(samples)
    test_num = floor(samples.shape[0] * ratio)
    error_num = 0
    trains = samples[test_num:, :]
    train_labels = labels[test_num:]
    for i in range(test_num):
        predict = classify0(samples[i, :], trains, train_labels, k)
        if predict != labels[i]:
            error_num += 1

    return error_num / test_num

def classify_person(filename='data/Ch02/datingTestSet2.txt', k = 3):
    results = ['', 'not at all', 'in small doses', 'in large doses']
    trains, train_labels = file2matrix(filename)
    trains, ranges, min_values = normalize(trains)
    gaming = float(input('Percent of time spent on video games: '))
    ffmiles = float(input('ff miles earned per year: '))
    ice_cream = float(input('liters of ice creams consumed per year: '))
    x = (np.array([ffmiles, gaming, ice_cream]) - min_values) / ranges
    predict = results[int(classify0(x, trains, train_labels, k))]
    print('you will probably like the person:', predict)


def image2vector(filename):
    vec = np.zeros(1024)
    idx = 0
    with open(filename, 'r') as f:
        for line in f:
            vec[idx:idx+32] = list(line[:-1])
            idx += 32

    return vec


def handwriting_classify_test(train='data/Ch02/digits/trainingDigits',
                              test='data/Ch02/digits/testDigits',
                              k=3):
    train_files = listdir(train)
    test_files = listdir(test)
    train_size = len(train_files)
    test_size = len(test_files)
    test_errors = 0

    # construct the train set and labels
    train_set = np.zeros((train_size, 1024))
    train_labels = np.zeros(train_size, dtype=int)
    idx = 0
    for file in train_files:
        train_labels[idx] = int(file.split('_')[0])
        train_set[idx, :] = image2vector(train + '/' + file)
        idx += 1

    # construct the test set and labels
    test_set = np.zeros((test_size, 1024))
    test_labels = np.zeros(test_size, dtype=int)
    idx = 0
    for file in test_files:
        test_labels[idx] = int(file.split('_')[0])
        test_set[idx, :] = image2vector(test + '/' + file)
        idx += 1

    # test the classifier
    for idx in range(test_size):
        predict = classify0(test_set[idx, :], train_set, train_labels, k)
        if predict != test_labels[idx]:
            test_errors += 1

    return test_errors / test_size


def run_classify(filename):
    pass


if __name__ == '__main__':
    from sys import argv
    if len(argv) == 1:
        # run simple test
        import unittest as ut

        class Test(ut.TestCase):
            def setUp(self):
                self.samples, self.labels = create_data_set()
                self.testfile = './kNN_testdata'

            def test_classify0(self):
                k = 2
                self.assertEqual('A', classify0([0.1, 0.2], self.samples, self.labels, k))
                self.assertEqual('B', classify0([1.2, 0.9], self.samples, self.labels, k))

            def test_file2matrix(self):
                samples, labels = file2matrix(self.testfile)
                self.assertEqual((3, 3), samples.shape)
                self.assertEqual((3,), labels.shape)
                self.assertTrue(np.all(samples == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
                self.assertTrue(np.all(labels == ['11', '12', '13']))

            def test_normalize(self):
                samples = np.arange(11)
                norm_samples, ranges, min_values = normalize(samples)
                self.assertTrue(np.all(norm_samples - np.arange(0.0, 1.01, 0.1) < 0.00001))
                self.assertEqual(10, ranges)
                self.assertEqual(0, min_values)

        ut.main()

    else:
        print('dating person error rate:', dating_classify_test())
        print('handwriting digits error rate:', handwriting_classify_test())
