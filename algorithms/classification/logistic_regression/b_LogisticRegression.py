import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from sklearn.preprocessing import LabelEncoder

#Notes for myself:
#binary logistic regression; here's everything you need to know
#1. https://statacumen.com/teach/SC1/SC1_11_LogisticRegression.pdf


class b_LogisticRegression:
  def __init__(self, lam=0, epsilon=1e-4, pgtol=1e-7, n_iters=1e6, solver='lbfgs'):
    self.epsilon = epsilon
    self.lam = lam
    self.n_iters = n_iters
    self.pgtol = pgtol
    self.solver = solver

  def sigmoid(self, x):
    return 1/(1+np.exp(-x))

  #function to minimize (used for lbfgs)
  def f(self, W, X, y, lam):
    Wx = np.dot(X, W)
    #np.exp(W)  np.exp(x)
    likelihood = (y-1)*Wx - np.log(1 + np.exp(-Wx))
    return -np.sum(likelihood)

  #derivative of function to minimize (used for lbfgs)
  def fprime(self, W, X, y, lam):
    Wx = np.dot(X, W)
    dl = np.dot(X.T, (y - self.sigmoid(Wx)))
    return -dl


  def gradient(self, W, X, y, lam):
    Wx = np.dot(X, W)
    dl = np.dot(X.T, (y - self.sigmoid(Wx)))
    return dl

  def hessian(self, W, X):
    Wx = np.dot(X, W)
    p = self.sigmoid(Wx)
    w = p * (1 - p)
    w = np.array([w]).T
    XTw = (X * w).T
    return -np.dot(XTw, X)

  #fit the data
  def fit(self, X, y):
    #turn labels into something we can use
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    n_classes = len(le.classes_)

    self.le = le
    self.classes_ = le.classes_
    self.n_classes_ = n_classes
    self.n_features_ = len(X[0])
    self.X = X
    self.y = y

    self.W = np.zeros(self.n_features_)  # initialize W 0

    print("solving with {}...".format(self.solver))
    #use lbfgs solver
    if self.solver=='lbfgs': 
      self.W, _, _ = scipy.optimize.fmin_l_bfgs_b(\
        self.f, self.W, self.fprime, args = (self.X, self.y, self.lam),\
        epsilon=self.epsilon, pgtol=self.pgtol) #Didn't work...? why
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
    p_0 = self.sigmoid(np.dot(X, W))
    while iters < n_iters:

      n_iters += 1

      #Do Newton-Raphson update
      grad = self.gradient(W, X, y, lam)
      hess = self.hessian(W, X)
      update = np.dot(np.linalg.inv(hess), grad)
      W = W - update

      #calculate error for convergence as the difference in probability
      p_new = self.sigmoid(np.dot(X, W))
      error = p_0 - p_new
      if np.min(np.abs(error)) < pgtol:
        print("Reached convergence")
        return W

      #Set new p_0
      p_0 = p_new

    print("Reached maximum number of iterations")
    return W

  #simple implementation of gradient descent (not stochastic)
  def solve_gradient_descent(self, W, X, y, lam, n_iters, pgtol, epsilon):    
    iters = 0
    while iters < n_iters:
      iters += 1
      Wx = np.dot(X, W)
      grad = np.dot(X, (y - sigmoid(Wx)) )
      W -= epsilon * (grad)

      if np.min(np.abs(grad)) < pgtol:
        print("Reached convergence")
        return 0
    print("Reached maximum number of iterations")
    return 1

  #return probabilities for tags given X
  def predict_proba(self, X):
    return self.sigmoid(np.dot(X, self.W))

  #return a list of predictions
  def predict(self, X):
    y = []
    for p in self.predict_proba(X):
      if p > 0.5:
        y.append(1)
      else:
        y.append(0)
    return self.le.inverse_transform(y)






