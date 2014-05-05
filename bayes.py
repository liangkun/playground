#!/usr/bin/env python3

import numpy as np
import re
import random
import feedparser
from operator import itemgetter


def create_data_set():
    docs = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
            ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
            ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
            ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels = [0, 1, 0, 1, 0, 1]

    return docs, labels


def create_vocab_list(data_set):
    vocabs = set()
    for data in data_set:
        vocabs |= set(data)

    return sorted(vocabs)


def words_to_vec(vocab_list, inputs):
    result = [0] * len(vocab_list)
    for word in inputs:
        if word in vocab_list:
            result[vocab_list.index(word)] += 1

    return result


def train_nb0(train_mat, train_cat):
    """Training naive bayes classifier.
    """
    num_words = len(train_mat[0])
    num_docs = len(train_mat)
    pp = np.sum(train_cat) / num_docs
    p0_words = np.ones(num_words)
    p0_word_count = num_words
    p1_words = np.ones(num_words)
    p1_word_count = num_words

    for sample, label in zip(train_mat, train_cat):
        if label == 0:
            p0_words += sample
            p0_word_count += np.sum(sample)
        else:
            p1_words += sample
            p1_word_count += np.sum(sample)

    return pp, np.log(p0_words/p0_word_count), np.log(p1_words/p1_word_count)


def classify_nb0(input, p0w, p1w, pp):
    p1 = np.sum(p1w * input) + np.log(pp)
    p0 = np.sum(p0w * input) + np.log(1 - pp)
    if p1 > p0: return 1
    else: return 0


def test_nb0_classifier():
    docs, lables = create_data_set()
    vocab_list = create_vocab_list(docs)
    train_mat = []
    for doc in docs:
        train_mat.append(words_to_vec(vocab_list, doc))
    pp, p0w, p1w = train_nb0(train_mat, lables)

    test_entry = ['love', 'my', 'dog']
    test_vec = words_to_vec(vocab_list, test_entry)
    label = classify_nb0(test_vec, p0w, p1w, pp)
    print('{} classified as {}'.format(test_entry, label))

    test_entry = ['garbage', 'dog', 'my', 'garbage']
    test_vec = words_to_vec(vocab_list, test_entry)
    label = classify_nb0(test_vec, p0w, p1w, pp)
    print('{} classified as {}'.format(test_entry, label))


def text_parse(text):
    splitter = re.compile(r'\W+')
    return [tok.lower() for tok in splitter.split(text) if len(tok) > 2]


def spam_test():
    docs = []
    labels = []
    for i in range(1, 26):
        with open('data/Ch04/email/spam/{}.txt'.format(i), 'r', encoding='ISO8859') as f:
            docs.append(text_parse(f.read()))
            labels.append(1)
        with open('data/Ch04/email/ham/{}.txt'.format(i), 'r', encoding='ISO8859') as f:
            docs.append(text_parse(f.read()))
            labels.append(0)

    vocab_list = create_vocab_list(docs)
    train_set = list(range(len(docs)))
    test_set = []
    test_set_count = int(len(docs) * 0.2)
    for i in range(test_set_count):
        rand_idx = random.randint(0, len(train_set)-1)
        test_set.append(train_set[rand_idx])
        del train_set[rand_idx]

    train_mat = []
    train_labels = []
    for idx in train_set:
        train_mat.append(words_to_vec(vocab_list, docs[idx]))
        train_labels.append(labels[idx])

    pp, p0w, p1w = train_nb0(train_mat, train_labels)

    error_count = 0
    for idx in test_set:
        if classify_nb0(words_to_vec(vocab_list, docs[idx]), p0w, p1w, pp) != labels[idx]:
            print('miss classify {} to {}'.format(labels[idx], 1-labels[idx]))
            error_count += 1

    print('Error rate on testing set is %{}'.format(error_count / test_set_count * 100))


def get_stop_words(fname):
    with open(fname, 'r') as f:
        return set(text_parse(f.read()))


def most_freq_words(docs, n, stop_words):
    words_count = {}
    for doc in docs:
        for word in doc:
            if word not in stop_words:
                words_count[word] = words_count.get(word, 0) + 1
    return sorted(words_count.items(), key=itemgetter(1), reverse=True)[:n]


def local_words(feed0, feed1):
    docs = []
    labels = []

    min_fnum = min(len(feed0['entries']), len(feed1['entries']))
    for i in range(min_fnum):
        docs.append(text_parse(feed0['entries'][i]['summary']))
        labels.append(0)
        docs.append(text_parse(feed1['entries'][i]['summary']))
        labels.append(1)

    vocab_list = create_vocab_list(docs)
    stop_words = get_stop_words('stopwords.txt')
    for word in stop_words:
        if word in vocab_list:
            vocab_list.remove(word)
    freq_words = most_freq_words(docs, 15, stop_words)
    for word in freq_words:
        if word[0] in vocab_list:
            vocab_list.remove(word[0])

    train_set = list(range(2 * min_fnum))
    test_set = []
    test_set_count = int(len(docs) * 0.1)
    for i in range(test_set_count):
        rand_idx = random.randint(0, len(train_set) - 1)
        test_set.append(train_set[rand_idx])
        del train_set[rand_idx]

    train_mat = []
    train_label = []
    for idx in train_set:
        train_mat.append(words_to_vec(vocab_list, docs[idx]))
        train_label.append(labels[idx])

    pp, p0w, p1w = train_nb0(train_mat, train_label)
    error_count = 0
    for idx in test_set:
        if classify_nb0(words_to_vec(vocab_list, docs[idx]), p0w, p1w, pp) != labels[idx]:
            error_count += 1

    print("Error rate on testing set is %{}".format(error_count / test_set_count * 100))

    return vocab_list, p0w, p1w


def get_top_words(sf, ny):
    vocab_list, p_sf, p_ny = local_words(sf, ny)
    top_sf = []
    top_ny = []
    for i in range(len(p_sf)):
        if p_sf[i] > -6.0: top_sf.append((vocab_list[i], p_sf[i]))
        if p_ny[i] > -6.0: top_ny.append((vocab_list[i], p_ny[i]))
    top_sf.sort(key=itemgetter(1), reverse=True)
    top_ny.sort(key=itemgetter(1), reverse=True)

    print('SF**'*8)
    for item in top_sf: print(item[0])

    print('NY**'*8)
    for item in top_ny: print(item[0])


if __name__ == '__main__':
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    get_top_words(sf, ny)