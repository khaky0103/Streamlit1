import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Tuple, Callable
from knn import parse_data, create_folds, create_train_test, \
    evaluation_metric, build_knn, create_train_test_less_than_index
import math
import random

# Title & load data
st.title('Predict Compressive Strength with k-NN')
data_load_state = st.info('Loading data...')
data = parse_data("concrete_compressive_strength.csv")
raw_column_names = ["Cement", "Slag", "Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate",
                    "Age", "Compressive Strength"]
data_df = pd.DataFrame(data=data, columns=raw_column_names)

# Find minimum and maximum for each independent variable

min_max = {}
column_names = ["Cement", "Slag", "Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age"]
for i in range(len(column_names)):
    min_max[column_names[i]] = (math.floor(data_df.iloc[:, i].min()), math.floor(data_df.iloc[:, i].max()))
data_load_state.success("Concrete Compressive Strength Data Loaded")

if st.checkbox("Show Data"):
    st.subheader('Data:')
    st.write(data_df)

# Pick numbers for independent variables (instead of min, use 0 for minimum)
st.subheader("Pick numbers for independent variables:")
cement = st.slider("Cement (kg in a cubic meter mixture)", 0, min_max["Cement"][1])
slag = st.slider("Slag (kg in a cubic meter mixture)", 0, min_max["Slag"][1])
ash = st.slider("Ash (kg in a cubic meter mixture)", 0, min_max["Ash"][1])
water = st.slider("Water (kg in a cubic meter mixture)", 0, min_max["Water"][1])
superplasticizer = st.slider("Superplasticizer (kg in a cubic meter mixture)",
                             0, min_max["Superplasticizer"][1])
coarse_aggregate = st.slider("Coarse Aggregate (kg in a cubic meter mixture)",
                             0, min_max["Coarse Aggregate"][1])
fine_aggregate = st.slider("Fine Aggregate (kg in a cubic meter mixture)",
                           0, min_max["Fine Aggregate"][1])
age = st.slider("Age (days)", 0, min_max["Age"][1])

# Show the values picked
st.subheader("Picked Values:")
picked_values = [cement, slag, ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]
configured_values_data = np.array([picked_values])
configured_values_df = pd.DataFrame(configured_values_data, columns=column_names)
st.table(configured_values_df)
# create folds and train and test sets (train with all data, test set is empty)
folds = create_folds(data, 10)
train, _ = create_train_test_less_than_index(folds, len(folds))

# Pick number of neighbors for k-nn
st.subheader("Pick Number of Nearest Neighbors to use in Prediction:")
k = st.slider("k", 1, 20)

# Find predicted strength
knn = build_knn(train)
predicted_strength = knn(k, picked_values)

# Show predicted strength
if st.button("Predict Compressive Strength"):
    st.info("Predicted Concrete compressive Strength: %.2f MPa" % predicted_strength)

# Show source code
if st.button("See Code"):
    st.subheader('Code:')
    with st.echo():
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
            for i in range(len(obs1) - 1):
                distance += (obs1[i] - obs2[i]) ** 2
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
                avg_value_for_regression = sum_k_nearest_neighbor_distance / k
                return avg_value_for_regression

            return knn


        def mean_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
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
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
            return m, (m - h, m + h)


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
            mean_error = sum_error / float(len(actual))
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
            mean_y = sum(train_y) / len(train_y)
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

        # MAIN
        # create folds and train and test sets (train with all data, test set is empty)
        folds = create_folds(data, 10)
        train, _ = create_train_test_less_than_index(folds, len(folds))
        # Find predicted strength
        knn = build_knn(train)
        predicted_strength = knn(k, picked_values)





