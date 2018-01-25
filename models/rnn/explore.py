import itertools

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from RNNNumpy import RNNNumpy

vocab_size = 5000
unknown_token = "UNKNOWN_TOKEN"

df = pd.read_csv('amazon_cells_labeled.csv')

df['sentence'] = df['sentence'].map(lambda x: word_tokenize(x.lower()))
word_freq = nltk.FreqDist(itertools.chain(*df['sentence']))
vocab = word_freq.most_common(vocab_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])


X = [[word_to_index[w] for w in [unknown_token] + sentence] for sentence in df['sentence']]
y = [[word_to_index[w] for w in sentence + [unknown_token]] for sentence in df['sentence']]



rnn = RNNNumpy(vocab_size)
o, s = rnn.forward_propagation(X[0])

preds = rnn.predict(X[0])
rnn.calculate_loss(X, y)