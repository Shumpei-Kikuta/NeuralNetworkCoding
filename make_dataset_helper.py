"""
Create the dataset 

Input: None
Output: X_train, X_test, y_train, y_test, n_labels
"""

import numpy as np 
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data


def prepare_iris():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    X_train = X_train.T
    X_test = X_test.T
    n_labels = len(np.unique(y))
    y_train = np.eye(n_labels)[y_train].T
    y_test = np.eye(n_labels)[y_test].T
    return X_train, X_test, y_train, y_test, n_labels


def prepare_digits():
    digit = load_digits()
    X, y = digit.data, digit.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    X_train = X_train.T
    X_test = X_test.T
    n_labels = len(np.unique(y))
    y_train = np.eye(n_labels)[y_train].T
    y_test = np.eye(n_labels)[y_test].T
    return X_train, X_test, y_train, y_test, n_labels



def prepare_mnist():
    (pre_X_train, pre_y_train), (pre_X_test, pre_y_test) = load_data()
    X_train = pre_X_train.reshape(pre_X_train.shape[0], pre_X_train.shape[1] * pre_X_train.shape[2]).T
    X_test = pre_X_test.reshape(pre_X_test.shape[0], pre_X_test.shape[1] * pre_X_test.shape[2]).T
    n_labels = len(np.unique(pre_y_train))
    y_train = np.eye(n_labels)[pre_y_train].T
    y_test = np.eye(n_labels)[pre_y_test].T

    return X_train, X_test, y_train, y_test, n_labels
    