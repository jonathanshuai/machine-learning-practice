import numpy as np
import pandas as pd
import sqlite3 as sql
import scipy.optimize
import scipy.stats
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt

import time 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#Our logistic regression implementation
from n_LogisticRegression import n_LogisticRegression

#Dataset from https://www.kaggle.com/aniruddhaachar/audio-features/data

#Read in the data
t = time.time()
print("Loading...")
dinner_df = pd.read_csv("dinner_audio.csv")
party_df = pd.read_csv("party_audio.csv")
sleep_df = pd.read_csv("sleep_audio.csv")
workout_df = pd.read_csv("workout_audio.csv")
#database = sql.connect("database.sqlite")
print("Done! {}s".format(time.time() - t))

#Add labels to the data
dinner_df['label'] = pd.Series(["dinner"]*len(dinner_df))
party_df['label'] = pd.Series(["party"]*len(dinner_df))
sleep_df['label'] = pd.Series(["sleep"]*len(dinner_df))
workout_df['label'] = pd.Series(["workout"]*len(dinner_df))

#Make one big dataframe
df = pd.concat([dinner_df, party_df, workout_df, sleep_df])
df = df.dropna()

#get the features we want and set the label column name
features = ['mfcc', 'scem', 'scom', 'srom', 'sbwm', 'tempo', 'rmse']
label = 'label'

#Create a minimalist scatterplot matrix just for fun
def scatterplot_matrix(df, features, kwargs, label='label'):
  fig, axs = plt.subplots(len(features),len(features))
  for (i, f1) in enumerate(features):
    for (j, f2) in enumerate(features):
      ax = axs[i,j]
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.set_xticks([])
      ax.set_yticks([])
      if j == 0:
        ax.set_ylabel(f1)
      if i == 0:
        ax.set_xlabel(f2)
        ax.xaxis.set_label_position('top')
      if i == j:
        continue
      else:
        le = LabelEncoder()
        labels = le.fit_transform(df[label])
        #labels = le.transform(df[label])
        ax.scatter(df[f2], df[f1], c=labels, **kwargs)
  plt.tight_layout(pad=0)
  plt.show()


#use our simple scatterplot matrix function
kwargs = {'s':2, 'cmap':'winter'}
#scatterplot_matrix(df=df, features=features, kwargs=kwargs, label=label)


#Create the df that we want to try predicting on
df = pd.concat([dinner_df, workout_df])
df = df.dropna()

labels = df[label]
dff = df[features]

#normalize the huge features
dff = (dff - dff.mean()) / (dff.max() - dff.min())

#Train and test sets
X_train, X_test, y_train, y_test = train_test_split(dff, labels, test_size=0.3)

X = X_train.as_matrix()
y = y_train.as_matrix()
X_test = X_test.as_matrix()

b = np.ones(X.shape[0])
X = np.c_[b, X]
b = np.ones(X_test.shape[0])
X_test = np.c_[b, X_test] 


#multinomial regression
clf_multi = n_LogisticRegression(lam=1)
clf_multi.fit(X, y)
predictions = clf_multi.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(acc)

'''
#for testing

def softmax(x):
  exp = np.exp(x - np.max(x))
  if exp.ndim==1:
    sm = exp / np.sum(exp)
  else:
    sm = exp / np.array([np.sum(exp, axis=1)]).T
  return sm

def cross_entropy(s, l):
  return -np.sum(np.dot(np.log(s).T, l))

def d_cross_entropy(x,y):
  grad = softmax(x) - y
  return grad

def dd_cross_entropy(x):
  sT = softmax(x).T
  ss = np.zeros((len(sT), len(sT)))
  for i in range(len(sT)):
    for j in range(len(sT)):
      if i == j:
        ss[i][j] = np.dot(sT[i], 1-sT[j])
      else:
        ss[i][j] = -np.dot(sT[i], sT[j])
  return ss


le = LabelEncoder()
le.fit(y)
y_index = le.transform(y)
n_classes = len(le.classes_)
y = [[0]*n_classes for _ in y_index]

for i in range(len(y_index)):
  y[i][y_index[i]] = 1


y = np.array(y)

W = np.ones((len(X[1]), n_classes))

#Exploring
x = np.dot(X, W)
s = softmax(np.dot(X,W))
cross_entropy(softmax(np.dot(X, W)), y)
-np.dot(np.log(s).T, y).trace()

#Training
epsilon = 1e-2
W = np.ones((len(X[1]), n_classes))

grad = np.dot(X.T, d_cross_entropy(np.dot(X, W), y))
W += epsilon * np.dot(X.T, d_cross_entropy(np.dot(X, W), y))


W = np.ones(len(X[0]))
y = y_index

def sigmoid(x):
  return 1/(1+np.exp(-x))

Wx = np.dot(X, W)
dl = np.dot(X.T, (y-1) + (1-sigmoid(Wx)) )

Wx = np.dot(X, W)
p = sigmoid(Wx)
np.dot(p, 1-p) #Note that p is one dimensional here
np.dot(X.T, X)
'''

from b_LogisticRegression import b_LogisticRegression


clf = b_LogisticRegression(solver='lbfgs')
clf.fit(X, y)
p_lbfgs = clf.predict(X_test)
w_lbfgs = clf.W


clf = b_LogisticRegression(solver='newton-raphson')
clf.fit(X, y)
p_nr = clf.predict(X_test)
w_nr = clf.W


clf = b_LogisticRegression(solver='')
clf.fit(X, y)
p_d = clf.predict(X_test)
w_d = clf.W


##Testing
#y_test = le.transform(y_test)
#x = np.dot(X_test, W)
#softmax(x)