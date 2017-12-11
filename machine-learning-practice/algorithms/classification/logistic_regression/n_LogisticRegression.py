import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from sklearn.preprocessing import LabelEncoder

#Notes for myself:
#If you forgot how softmax regression works:
#1. https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
#2. https://deepnotes.io/softmax-crossentropy
#3. http://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html
#The bias weight is folded into W (expects X to have a column of 1's)


class n_LogisticRegression:
  def __init__(self, lam=0, epsilon=1e-4, pgtol=1e-7, n_iters=1e5, solver='lbfgs'):
    self.epsilon = epsilon
    self.lam = lam
    self.n_iters = n_iters
    self.pgtol = pgtol
    self.solver = 'lbfgs'

  #softmax or p(y=k|x); see notes for details
  def softmax(self, x):
    exp = np.exp(x - np.max(x))
    if exp.ndim == 1:
      sm = exp / np.sum(exp)
    else:
      sm = exp / np.array([np.sum(exp, axis=1)]).T
    return sm

  #Loss metric
  def cross_entropy(self, s, l):
    return - np.dot(np.log(s).T, l).trace()

  #derivative of cross entropy with respect to Wx (note: chain rule for  dWx/dW is applied outside)
  def d_cross_entropy(self, x, y):
    grad = self.softmax(x) - y
    return grad

  #function to minimize 
  def f(self, W, X, y, lam):
    W = np.array(W).reshape((self.n_features_, self.n_classes_))
    x = np.dot(X, W)
    s = self.softmax(x)
    reg = lam * np.sum(np.square(W))
    return self.cross_entropy(s, y) + reg

  #derivative of function to minimize
  def fprime(self, W, X, y, lam):
    W = np.array(W).reshape((self.n_features_, self.n_classes_))
    x = np.dot(X, W)
    grad = np.dot(X.T, self.d_cross_entropy(x, y)).ravel()
    reg = 2 * lam * np.sum(W)
    return grad + reg

  #fit the data
  def fit(self, X, y):
    #turn labels into something we can use
    le = LabelEncoder()
    le.fit(y)
    y_index = le.transform(y)
    n_classes = len(le.classes_)

    y = [[0]*n_classes for _ in y_index]

    for i in range(len(y_index)):
      y[i][y_index[i]] = 1

    self.le = le
    self.classes_ = le.classes_
    self.n_classes_ = n_classes
    self.n_features_ = len(X[0])
    self.X = X
    self.y = np.array(y)

    self.W = np.zeros((self.n_features_, self.n_classes_))  # initialize W 0

    #use lbfgs solver
    if self.solver=='lbfgs': 
      self.W, _, _ = scipy.optimize.fmin_l_bfgs_b(\
        self.f, self.W, self.fprime, args = (self.X, self.y, self.lam),\
        epsilon=self.epsilon, pgtol=self.pgtol) #Didn't work...? why
    else:
      self.solve_gradient_descent()
    
    self.W = self.W.reshape((self.n_features_, self.n_classes_))

  #simple implementation of gradient descent (not stochastic)
  def solve_gradient_descent(self):    
    iters = 0
    while iters < self.n_iters:
      iters += 1
      x = np.dot(self.X, self.W)
      grad = np.dot(self.X.T, self.d_cross_entropy(x, self.y))
      reg = 2 * lam * np.sum(W)
      self.W -= self.epsilon * (grad + reg)

      if np.min(np.abs(grad)) < self.pgtol:
        print("Reached convergence")
        return 0
    print("Reached maximum number of iterations")
    return 1

  #return probabilities for tags given X
  def predict_proba(self, X):
    return self.softmax(np.dot(X, self.W))

  #return a list of predictions
  def predict(self, X):
    y = []
    for p in self.predict_proba(X):
      y.append(np.argmax(p))
    return self.le.inverse_transform(y)




