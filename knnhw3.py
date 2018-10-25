import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kclass

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

    kn = kclass(n_neighbors=3)
    kn.fit(x_train, y_train)
    score = (1 - kn.score(x_test, y_test)) * 100
    print('Classification error n=3: {0:.4f}%'.format(score))

    kn = kclass(n_neighbors=5)
    kn.fit(x_train, y_train)
    score = (1 - kn.score(x_test, y_test)) * 100
    print('Classification error n=5: {0:.4f}%'.format(score))

    kn = kclass(n_neighbors=7)
    kn.fit(x_train, y_train)
    score = (1 - kn.score(x_test, y_test)) * 100
    print('Classification error n=7: {0:.4f}%'.format(score))

main()
