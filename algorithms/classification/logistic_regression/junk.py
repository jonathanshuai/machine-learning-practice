#for testing

le = LabelEncoder()
le.fit(y)
y_index = le.transform(y)
n_classes = len(le.classes_)-1
y = [[0]*n_classes for _ in y_index]

for i in range(len(y_index)):
  if y_index[i] < n_classes:
    y[i][y_index[i]] = 1


y = np.array(y)

#Exploring

def sigmoid(x):
  return 1/(1+np.exp(-x))

def probability(W, X):
  x = np.dot(X, W)
  if W.ndim == 1:
    return np.exp(x)/np.array([1 + np.sum(np.exp(x), axis=0)]).T
  else:
    return np.exp(x)/np.array([1 + np.sum(np.exp(x), axis=1)]).T

def gradient(W, X, y, lam):
  p = probability(W,X)
  return np.dot(X.T, y - p)

def predict_proba(W, X):
  p = probability(W, X)
  return np.c_[p,1-np.sum(p, axis=1)]

def predict(W, X):
  y = []
  for p in predict_proba(W, X):
    y.append(np.argmax(p))
  return y

epsilon = 1e-2

n_classes
n_features = len(X[0])

def hessian(W, X):
  kj = n_classes * n_features
  hess = np.zeros((kj, kj))
  p = probability(W, X)
  for j in range(n_classes):
    for k in range(n_classes):
      m = j*n_features
      n = k*n_features
      p_j = p[:,j]
      p_k = p[:,k]
      if j == k:
        pp = p_j * (1-p_k)
      else:
        pp = p_j * p_k
      pp = np.array([pp]).T
      XTw = (X * pp).T
      h_jk = np.dot(XTw, X)
      if j == k:
        h_jk = -h_jk
      hess[m:m+n_features, n:n+n_features] = h_jk
  return hess

def predict_proba(W, X):
  p = probability(W, X)
  return np.c_[p,1-np.sum(p, axis=1)]

#return a list of predictions
def predict(W, X):
  y = []
  for p in predict_proba(W, X):
    y.append(np.argmax(p))
  return y

def gradient2(W, X, y, lam):
  Wx = np.dot(X, W)
  dl = np.dot(X.T, (y-1) + (1-sigmoid(Wx)))
  return dl

def hessian2(W, X):
  Wx = np.dot(X, W)
  p = sigmoid(Wx)
  w = p * (1 - p)
  w = np.array([w]).T
  XTw = (X * w).T
  return -np.dot(XTw, X)



W = np.zeros((len(X[1]), n_classes))
p_0 = probability(W, X)

grad = gradient(W, X, y, 0).ravel()
hess = hessian(W, X)
update = np.dot(np.linalg.inv(hess), grad).reshape((len(X[1]), n_classes))
W = W - update
W
error = p_0 - probability(W,X)
np.min(np.abs(error))
p_0 = probability(W, X)


accuracy_score(predict(W, X_test), le.transform(y_test))



W = np.zeros((len(X[1]), n_classes))[:,0]

grad = clf.gradient(W, X, y_index, 0)
hess = clf.hessian(W, X)
update = np.dot(np.linalg.inv(hess), grad)
W = W - update
W


grad = gradient(W, X, y, 0)
W = W + grad
W



predict(W, X_test)
le.transform(y_test)
accuracy_score(predict(W, X_test), le.transform(y_test))









def gradient(W, X, y, lam):
  Wx = np.dot(X, W)
  dl = np.dot(X.T, (y-1) + (1-sigmoid(Wx)))
  return dl

def hessian(W, X):
  Wx = np.dot(X, W)
  p = sigmoid(Wx)
  w = p * (1 - p)
  XTw = (X * w).T
  return -np.dot(XTw, X)

W = np.ones((len(X[1]), n_classes))

grad = gradient(W, X, y, 0)
hess = hessian(W, X)
hess_inv = np.linalg.inv(hess)
W = W - np.dot(hess_inv, grad)
W

predict(W, X_test)
le.transform(y_test)
accuracy_score(predict(W, X_test), le.transform(y_test))






