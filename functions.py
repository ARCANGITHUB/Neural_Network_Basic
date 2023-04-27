import numpy as np
import pandas as pd

rg = np.random.default_rng()


def generate_data(n_features, n_values):
    features = rg.random((n_features, n_values))
    weights = rg.random((1, n_values))[0]   # One dimensional set of weights
    targets = np.random.choice([0, 1], n_features)
    data = pd.DataFrame(features, columns=['X0', 'X1', 'X2'])
    data["targets"] = targets
    return data, weights


def get_weighted_sum(feature, weights, bias):
    return np.dot(feature, weights) + bias


def sigmoid(w_sum):
    return 1/(1+np.exp(-w_sum))


def cross_entropy(target, prediction):
    return -(target*np.log10(prediction) + (1-target)*np.log10(1-prediction))


def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x, w in zip(feature, weights):
        new_w = w + l_rate*(target-prediction)*x
        new_weights.append(new_w)
    return new_weights


def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate*(target-prediction)