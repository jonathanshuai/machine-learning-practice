import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets

from OneLayerNeuralNet import OneLayerNeuralNet

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = (X - np.min(X)) / (np.max(X) - np.min(X))

olnn = OneLayerNeuralNet()
model = olnn.train(X, y)
preds = olnn.predict(X)
probs = olnn.predict_proba(X)
probs[np.arange(len(X)), y]
probs[:,y]