import random
from typing import List, Dict, Tuple, Callable
import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import streamlit as st

@st.cache
def parse_data(file_name: str) -> List[List]:
    '''
    The function parse_data loads the data from the specified file and returns a List of Lists.
    The outer List is the data set and each element (List) is a specific observation.
    Each value of an observation is for a particular measurement. This is what we mean by "tidy" data.
    '''
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data

def create_folds(xs: List, n: int) -> List[List[List]]:
    '''
    The function create_folds will take a list (xs) and split it into n equal folds with each fold
    containing one-tenth of the observations.
    '''
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def create_train_test(folds: List[List[List]], index: int) -> Tuple[List[List], List[List]]:
    """ Function that take our n folds and return the train and test sets. """
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i == index:
            test = fold
        else:
            training = training + fold
    return training, test


def euclidean_distance(obs1, obs2):
    '''
    This function calculates the euclidean distance between obs1 and obs2.

    Parameters:
    - obs1: observation 1
    - obs2: observation 2

    Returns: The euclidean distance
    '''
    distance = 0.0
    for i in range(len(obs1) -1):
        distance += (obs1[i] - obs2[i])**2
    return math.sqrt(distance)


def build_knn(database):
    """
    This function builds the knn model.

    Parameters:
    - database: observation 1

    Returns: knn function which takes in a parameter k which is the number of neighbors it will take,
    and query which is the observation it will be testing against
    """
    def knn(k, query):
        distances = []
        for i in range(len(database)):
            distances.append((database[i], euclidean_distance(database[i], query)))
        distances.sort(key=lambda tup: tup[1])
        k_nearest_neighbors = distances[:k]
        sum_k_nearest_neighbor_distance = 0
        for neighbor in k_nearest_neighbors:
            sum_k_nearest_neighbor_distance += neighbor[0][8]
        avg_value_for_regression = sum_k_nearest_neighbor_distance/k
        return avg_value_for_regression
    return knn


def mean_confidence_interval(data: List[float], confidence: float=0.95) -> Tuple[float, Tuple[float, float]]:
    """
    This function calculates the mean of data and the confidence bounds of that mean.

    Parameters:
    - data:  a List of floating point numbers representing the data.
    - confidence: a scalar (float) indicating the desired confidence bounds. The default is 95% (0.95).

    Returns: the mean and a Tuple of the (low, high) values of the confidence% bounds. For example, 23.8 (21.2, 26.4)
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, (m-h, m+h)


def mean_squared_error(actual, predicted):
    """
    This calculates the mean squared error of two sets of data.

    Parameters:
    - actual: actual values
    - predicted: predicted values

    Returns: The mean squared error
    """
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += ((predicted[i] - actual[i]) ** 2)
    mean_error = sum_error/float(len(actual))
    return mean_error


def null_model(train):
    """
    This function generates the null model, which is the mean of the train data.

    Parameters:
    - train: the training data

    Returns: The mean concrete compressive strength of the train data
    """
    train_y = []
    for i in range(len(train)):
        train_y.append(train[i][8])
    mean_y = sum(train_y)/len(train_y)
    return mean_y


def evaluation_metric(actual: List, predicted: List) -> float:
    """ This function sets the evaluation metric as the mean squared error."""
    return mean_squared_error(actual, predicted)


def create_train_test_less_than_index(folds: List[List[List]], index: int) -> Tuple[List[List], List[List]]:
    """
    This function generates the train and test data from the fold and an index.

    Parameters:
    - folds: folds of the data
    - index: number of folds to go up to for the training data

    Returns: training and testing set based on above specifications
    """
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i <= index:
            training = training + fold
        else:
            test = test + fold
    return training, test





