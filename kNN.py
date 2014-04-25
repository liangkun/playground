#!/usr/bin/env python3

import numpy as np
from operator import itemgetter


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

    # get he neighber's votes
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

        ut.main()

    else:
        run_classify(argv[1])
