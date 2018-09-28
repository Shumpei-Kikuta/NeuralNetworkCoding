import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def prepare_dataset():
    iris = load_iris()
    X, y = iris.data[:100], iris.target[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.reshape(1, y_train.shape[0])
    y_test = y_test.reshape(1, y_test.shape[0])
    return X_train, X_test, y_train, y_test