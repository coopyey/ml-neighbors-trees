import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier as tclass

def read_data():
    tr_data = pd.read_csv('zip.train', header=None, delim_whitespace=True, skipinitialspace=True)
    te_data = pd.read_csv('zip.test', header=None, delim_whitespace=True, skipinitialspace=True)

    #Set training data
    x_tr = tr_data.loc[:, tr_data.columns != 0]
    y_tr = tr_data.loc[:, tr_data.columns == 0]

    #Set testing data
    x_te = te_data.loc[:, te_data.columns != 0]
    y_te = te_data.loc[:, te_data.columns == 0]

    return(x_tr, y_tr.values.ravel(), x_te, y_te.values.ravel())

def main():
    (x_train, y_train, x_test, y_test) = read_data()

    tree = tclass(min_samples_leaf=1)
    tree.fit(x_train, y_train)
    score = (1 - tree.score(x_test, y_test)) * 100
    print('Classification error leaf=1: {0:.4f}%'.format(score))

    tree = tclass(min_samples_leaf=5)
    tree.fit(x_train, y_train)
    score = (1 - tree.score(x_test, y_test)) * 100
    print('Classification error leaf=5: {0:.4f}%'.format(score))

    tree = tclass(min_samples_leaf=6)
    tree.fit(x_train, y_train)
    score = (1 - tree.score(x_test, y_test)) * 100
    print('Classification error leaf=6: {0:.4f}%'.format(score))

    tree = tclass(min_samples_leaf=7)
    tree.fit(x_train, y_train)
    score = (1 - tree.score(x_test, y_test)) * 100
    print('Classification error leaf=7: {0:.4f}%'.format(score))

main()