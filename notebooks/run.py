from constants import *
from helpers import *
from models import *
from training import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse

reviews = load_dataset(DATA_FOLDER + 'Reviews.csv')

text = reviews['Text']
scores = reviews['Score']

try:
    X = scipy.sparse.load_npz('neuralnet2000.npz')
    print(f'X was read from file')
except:
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))

    X = vectorizer.fit_transform(text)
    print(f'X was vectorized')

    scipy.sparse.save_npz('neuralnet10000.npz', X)

X_train, X_test, y_train, y_test = train_test_split(X, scores, test_size=0.1, random_state=41)
print(f'X_train size is {X_train.shape}')
y_train_torch = torch.tensor(y_train.values).float()

remainder = X_train.shape[0] % batch_size

params = {'n':128, 'dropout_rate':0.3, 'size':X.shape[1]}
net, loss = train_net(WordNet, num_epochs, t.FloatTensor(X_train.A[:-remainder]), y_train_torch, nn.MSELoss(), params)
#net = WordNet(params)
#net.load_state_dict(t.load('wordNet.pt'))
#net.eval()

test_output = net(t.FloatTensor(X_test.A))

print(test_output.narrow(0,0,10))
print(y_test.values[:10])
measure_performance(test_output.detach().numpy(), y_test)

nb_test_errors = t.sum(t.round(test_output)!=(t.tensor(y_test.values).float()))
print(f'Errors: {nb_test_errors} out of {test_output.shape}')
