from models import *
from constants import *
from helpers import *

import scipy.sparse
import torch as t
import pickle
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model

def train_regression(vectorizer, text, scores, tfidf_path, model_path):
    X = vectorizer.fit_transform(text)
    scipy.sparse.save_npz(tfidf_path, X)

    X_train, X_test, y_train, y_test = train_test_split(X, scores, test_size=0.1, random_state=41)
    print(f'X_train shape: {X_train.shape}')

    try:
        regr = pickle.load(open(model_path, 'rb'))
    except:
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        pickle.dump(regr, open(model_path, 'wb'))

    y_pred = regr.predict(X_test)
    measure_performance(y_pred, y_test)

#training a model in batches
def train_net(model, epochs, X_train, y_train, criterion, params):
    '''Automated training of model using batch gradient descent. '''
    learning_rate = 1e-3
    net = model(params)
    optimizer = t.optim.Adam(net.parameters(), lr=learning_rate)
    losses = t.Tensor(epochs)

    for epoch in range(epochs):
        print(f'Starting epoch {epoch}')
        epoch_loss = 0
        for batch_index in range(0, X_train.size(0), batch_size):

            output = net(X_train.narrow(0, batch_index, batch_size))

            y = y_train.narrow(0, batch_index, batch_size)

            loss = criterion(output, y)
            epoch_loss += loss.item()
            losses[epoch] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Loss: {loss.item()}')
    t.save(net.state_dict(), "../models/wordNet.pt")
    return net, loss
