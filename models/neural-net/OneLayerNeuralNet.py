import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from sklearn.preprocessing import LabelEncoder

# Here are some useful links:
# https://www.youtube.com/watch?v=tIeHLnjs5U8


class OneLayerNeuralNet:
  def __init__(self, epsilon=1e-2, lam=1e-2, hidden_dims=4, n_iters=1e4):
    self.epsilon = epsilon
    self.lam = lam
    self.hidden_dims = hidden_dims
    self.n_iters = int(n_iters)

  def loss_function(self):
    W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
    probs = self.predict_proba(self.X)
    correct_logprobs = -np.log(probs[range(len(self.X)), self.y])
    data_loss = np.sum(correct_logprobs)
    data_loss += self.lam/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1/len(self.X) * data_loss

  def predict_proba(self, X):
    W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

  def predict(self, X):
    probs = self.predict_proba(X)
    W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']

    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    print(a1)
    return np.argmax(probs, axis=1)

  def train(self, X, y):
    self.X = X
    le = LabelEncoder()
    self.y = le.fit_transform(y)
    self.output_dims = len(le.classes_)
    self.input_dims = X.shape[1]
    
    np.random.seed(0)
    W1 = np.random.randn(self.input_dims, self.hidden_dims) / np.sqrt(self.input_dims)
    b1 = np.zeros((1, self.hidden_dims))
    W2 = np.random.randn(self.hidden_dims, self.output_dims) / np.sqrt(self.hidden_dims)
    b2 = np.zeros((1, self.output_dims))

    self.model = {'W1': W1, 'b1': b1, 'W2' : W2, 'b2' : b2}
    for i in range(0, self.n_iters):
      probs = self.predict_proba(self.X)
    
      z1 = self.X.dot(W1) + b1
      a1 = np.tanh(z1)
    
      delta3 = probs
      delta3[range(len(self.X)), y] -= 1
      dW2 = (a1.T).dot(delta3)
      db2 = np.sum(delta3, axis=0, keepdims=True)
      delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
      dW1 = np.dot(self.X.T, delta2)
      db1 = np.sum(delta2, axis=0)

      #dW2 += self.lam * W2
      #dW1 += self.lam * W1

      W1 -= self.epsilon * dW1
      b1 -= self.epsilon * db1
      W2 -= self.epsilon * dW2
      b2 -= self.epsilon * db2


      self.model = {'W1': W1, 'b1': b1, 'W2' : W2, 'b2' : b2}

      if i % 1000 == 0:
        print("Loss after iteration {}: {}".format(i, self.loss_function()))
    return self.model


      