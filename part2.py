from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset,k):
    data, labels , t_centers = dataset

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Fit KMeans clustering
    kmeans = KMeans(n_clusters=k, init='random', random_state=42)
    kmeans.fit(scaled_data)

    centers = kmeans.cluster_centers_
    
    distances = np.sqrt(np.sum((scaled_data - centers[kmeans.labels_])**2, axis=1))
    
    sse = np.sum(distances ** 2)
    
    return sse

def fit_kmeans_inertia(dataset,k):
    data, labels , centers = dataset

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=k, init='random', random_state=42)
    kmeans.fit(scaled_data)

    inertia = kmeans.inertia_
  
    return inertia


def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays

    b = datasets.make_blobs(center_box=(-20,20), n_samples=20, centers=5, random_state=12,return_centers=True)
    X, y,returned_center = b
    dct = answers["2A: blob"] = [X, y,returned_center]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    
    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair   
    
    sse_values = []
    for k in range(1, 8 + 1):
        sse = fit_kmeans(b, k)
        sse_values.append((k, sse))

    sse_values_final = [[k, float(sse)] for k, sse in sse_values]

    dct = answers["2C: SSE plot"] = sse_values_final

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    # dct value has the same structure as in 2C

    inertia_values = []
    for k in range(1, 8 + 1):
        inertia = fit_kmeans_inertia(b, k)
        inertia_values.append((k, inertia))

    inertia_values_final = [[k, float(inertia)] for k, inertia in inertia_values]

    dct = answers["2D: inertia plot"] = inertia_values_final
 

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
