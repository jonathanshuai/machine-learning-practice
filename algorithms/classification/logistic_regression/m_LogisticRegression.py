import warnings
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from sklearn.preprocessing import LabelEncoder

#Notes for myself:
#Let's talk about maximum likelihood estimator for multinomial logistic regression
#Here's everything you need to know:
#1. https://czep.net/stat/mlelr.pdf 


class m_LogisticRegression:
  def __init__(self, lam=0, epsilon=1e-4, pgtol=1e-7, n_iters=1e6, solver='lbfgs'):
    self.epsilon = epsilon
    self.lam = lam
    self.n_iters = n_iters
    self.pgtol = pgtol
    self.solver = solver
 
  #Probability (like softmax but we saved a class using 1 + exp(x))
  def probability(self, W, X):
    x = np.dot(X, W)
    if W.ndim == 1:
      return np.exp(x)/np.array([1 + np.sum(np.exp(x), axis=0)]).T
    else:
      return np.exp(x)/np.array([1 + np.sum(np.exp(x), axis=1)]).T

  #log-likelihood function that we want to maximize
  def loglikelihood(self, W, X, y):
    x = np.dot(X, W)
    yWx = y * x
    a = np.log(1 + np.sum(np.exp(x), axis=1))
    a = np.array([a]).T
    loglikelihood = np.sum(yWx - a)
    return loglikelihood

  #first derivative of log likelihood
  def gradient(self, W, X, y):
    p = self.probability(W,X)
    grad = np.dot(X.T, y - p)
    return grad

  #second derivative of log likelihood. 
  #the hessian is a k x p matrix with k number of classes and p number of features
  #here we calculated each p x p block for every permutation of 
  #cols (correspoing to classes) of W
  #Note: this can be optimized (look at sklearn's fast hessian)
  def hessian(self, W, X):
    kp = self.n_classes_ * self.n_features_
    hess = np.zeros((kp, kp))
    p = self.probability(W, X)
    #print(W)
    for j in range(self.n_classes_):
      for i in range(self.n_classes_):
        m = j*self.n_features_
        n = i*self.n_features_
        p_j = p[:,j]
        p_i = p[:,i]
        if j == i:
          pp = p_j * (1-p_i)
        else:
          pp = p_j * p_i
        pp = np.array([pp]).T
        XTw = (X * pp).T
        h_ji = np.dot(XTw, X)
        if j == i:
          h_ji = -h_ji
        hess[m:m+self.n_features_, n:n+self.n_features_] = h_ji
    return hess

  #function used in lbfgs
  def f(self, W, X, y, lam):
    W = W.reshape((self.n_features_, self.n_classes_))
    reg = lam * np.sum(np.square(W))
    return -self.loglikelihood(W, X, y) + reg

  #derivative used in lbfgs
  def fprime(self, W, X, y, lam):
    W = W.reshape((self.n_features_, self.n_classes_))
    grad = self.gradient(W, X, y)
    reg = 2 * lam * np.sum(W)
    W = grad + reg
    return -W.ravel()

  #fit the data
  def fit(self, X, y):
    #turn labels into something we can use
    le = LabelEncoder()
    le.fit(y)
    y_index = le.transform(y)
    n_classes = len(le.classes_) - 1

    y = [[0]*n_classes for _ in y_index]

    for i in range(len(y_index)):
      if y_index[i] < n_classes:
        y[i][y_index[i]] = 1

    self.le = le
    self.classes_ = le.classes_
    self.n_classes_ = n_classes
    self.n_features_ = len(X[0])
    self.X = X
    self.y = np.array(y)

    self.W = np.zeros((self.n_features_, self.n_classes_))  # initialize W 0

    print("solving with {}...".format(self.solver))
    #use lbfgs solver
    if self.solver=='lbfgs': 
      self.W, _, _ = scipy.optimize.fmin_l_bfgs_b(\
        self.f, self.W, self.fprime, args = (self.X, self.y, self.lam),\
        epsilon=self.epsilon, pgtol=self.pgtol)
      self.W = self.W.reshape((self.n_features_, self.n_classes_))
    #use newton-raphson solver
    elif self.solver=='newton-raphson':
      self.W = self.solve_newton_raphson(self.W, self.X, self.y, self.lam, \
        self.n_iters, self.pgtol)
    #use simple gradient descent 
    else:
      self.W = self.solve_gradient_descent(self.W, self.X, self.y, self.lam, \
        self.n_iters, self.pgtol, self.epsilon)
    
  #use by specifying "newton-raphson" for solver
  def solve_newton_raphson(self, W, X, y, lam, n_iters, pgtol):
    iters = 0
    p_0 = self.probability(W, X)
    while iters < n_iters:
      n_iters += 1

      #perform the newton update
      grad = self.gradient(W, X, y).ravel()
      hess = self.hessian(W, X)
      update = np.dot(np.linalg.inv(hess), (grad))
      update = update.reshape(self.n_features_, self.n_classes_)
      W = W - update

      #get error for convergence as the difference in probability
      p_new = self.probability(W, X)
      error = p_0 - p_new
      if np.min(np.abs(error)) < pgtol:
        print("Reached convergence")
        return W

      #set the new p_0
      p_0 = p_new

    print("Reached maximum number of iterations")
    return W

  #simple implementation of gradient descent (not stochastic)
  def solve_gradient_descent(self, W, X, y, lam, n_iters, pgtol, epsilon):   
    iters = 0
    while iters < n_iters:
      iters += 1
      grad = self.gradient(W, X, y)
      reg = 2 * lam * np.sum(W)
      W -= epsilon * (grad + reg)

      if np.min(np.abs(grad)) < pgtol:
        print("Reached convergence")
        return 0
    print("Reached maximum number of iterations")
    return 1

  #return probabilities for tags given X
  def predict_proba(self, X):
    p = self.probability(self.W, X)
    return np.c_[p,1-np.sum(p, axis=1)]

  #return a list of predictions
  def predict(self, X):
    y = []
    for p in self.predict_proba(X):
      y.append(np.argmax(p))
    return self.le.inverse_transform(y)




