import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier as fclass

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

    forest = fclass(n_estimators=1)
    forest.fit(x_train, y_train)
    score = (1 - forest.score(x_test, y_test)) * 100
    print('Classification error leaf=1: {0:.4f}%'.format(score))

    forest = fclass(n_estimators=7)
    forest.fit(x_train, y_train)
    score = (1 - forest.score(x_test, y_test)) * 100
    print('Classification error leaf=7: {0:.4f}%'.format(score))

    forest = fclass(n_estimators=10)
    forest.fit(x_train, y_train)
    score = (1 - forest.score(x_test, y_test)) * 100
    print('Classification error leaf=10: {0:.4f}%'.format(score))

    forest = fclass(n_estimators=15)
    forest.fit(x_train, y_train)
    score = (1 - forest.score(x_test, y_test)) * 100
    print('Classification error leaf=15: {0:.4f}%'.format(score))

main()