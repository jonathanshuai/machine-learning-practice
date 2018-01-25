# Following along from http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from sklearn.preprocessing import LabelEncoder

class RNNNumpy:
  def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
    self.word_dim = word_dim
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate

    self.U = np.random.uniform(-np.sqrt(1/word_dim), np.sqrt(1/word_dim), (hidden_dim, word_dim))
    self.V = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (word_dim, hidden_dim))
    self.W = np.random.uniform(-np.sqrt(1/hidden_dim), np.sqrt(1/hidden_dim), (hidden_dim, hidden_dim))

def forward_propagation(self, x):
  T = len(x)
  s = np.zeros((T + 1, self.hidden_dim))
  s[-1] = np.zeros(self.hidden_dim) #What is this for??
  o = np.zeros((T, self.word_dim))

  for t in np.arange(T):
    s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
    o[t] = softmax(self.V.dot(s[t]))

  return [o, s]

RNNNumpy.forward_propagation = forward_propagation

def predict(self, x):
  o, s = self.forward_propagation(x)
  return np.argmax(o, axis=1)

RNNNumpy.predict = predict

def calculate_total_loss(self, x, y):
  L = 0
  for i in np.arange(len(y)):
    o, s = self.forward_propagation(x[i])
    correct_word_predictions = o[np.arange(len(y[i])), y[i]]
    L += -1 * np.sum(np.log(correct_word_predictions))
  return L

def calculate_loss(self, x, y):
  N = np.sum((len(y_i) for y_i in y))
  return self.calculate_total_loss(x, y) / N

RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss

def bptt(self, x, y):
  T = len(y)
  o, s = self.forward_propagation(x)
  dLdU = np.zeros(self.U.shape)
  dLdV = np.zeros(self.V.shape)
  dLdW = np.zeros(self.W.shape)
  delta_o = o
  delta_o[np.arange(len(y)), y] -= 1

  for t in np.arange(T)[::-1]:
    dLdV += np.outer(deta_o[t], s[t].T)
    # Initial delta calculation
    delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
    # Backpropagation through time (for at most self.bptt_truncate steps)
    for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
        # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
        dLdW += np.outer(delta_t, s[bptt_step-1])              
        dLdU[:,x[bptt_step]] += delta_t
        # Update delta for next step
        delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
  return [dLdU, dLdV, dLdW]

def softmax(X):
  denom = np.exp(X).sum()
  return np.exp(X)/denom