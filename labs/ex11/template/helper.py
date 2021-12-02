# -*- coding: utf-8 -*-
"""Some helper functions."""
import os
import shutil
import numpy as np
from matplotlib.pyplot import imread


def load_data():
    """Load data and convert it to the metrics system."""
    path_dataset = "faithful.csv"
    data = np.loadtxt(path_dataset, delimiter=" ", skiprows=0)
    return data


def normalize_data(data):
    """normalize the data by (x - mean(x)) / std(x)."""
    mean_data = np.mean(data, axis=0)
    data = data - mean_data
    std_data = np.std(data)
    data = data / std_data
    return data


def build_dir(dir):
    """build a new dir. if it exists, remove it and build a new one."""
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def load_image(path):
    """use the scipy.misc to load the image."""
    return imread(path)


def build_distance_matrix(data, mu):
    """Builds a distance matrix.

    Args:
        data: numpy array of shape = (N, d). Original data.
        mu: numpy array of shape = (k, d).
        Each row corresponds to a cluster center.

    Returns:
        numpy array of shape (N, k):
            squared distances matrix, the value row i column j corresponds to
            the squared distance of datapoint i with cluster center j.
    """
    n, k = data.shape[0], mu.shape[0]
    distances = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            distances[i, j] = np.linalg.norm(data[i, :] - mu[j, :]) ** 2
    return distances
