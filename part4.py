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

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(dataset, n_clusters, linkage='ward'):
    
    data, labels = dataset
        
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
        
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    hierarchical_cluster.fit(data_scaled)
        
    return hierarchical_cluster.labels_

def fit_modified(dataset, cutoff_distance, linkage_method):
    
    data, labels = dataset
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    Z = scipy_linkage(data_scaled, method=linkage_method)  
    
    num_clusters = len(np.where(Z[:, 2] > cutoff_distance)[0]) + 1
    
    hierarchical_cluster = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage_method)
    hierarchical_cluster.fit(data_scaled)
    
    return hierarchical_cluster.labels_


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {}
    
    n_samples = 100
    seed = 42

    nc_data,nc_labels = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)

    nm_data,nm_labels = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)

    b_data,b_labels = datasets.make_blobs(n_samples=n_samples, random_state=seed)

    # blobs with varied variances
    bvv_data,bvv_labels = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)

    # Anisotropicly distributed data
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    add_data = np.dot(X, transformation)
    add_labels = y 
    

    dct["nc"] = [nc_data, nc_labels]
    dct["nm"] = [nm_data, nm_labels]
    dct["bvv"] = [bvv_data, bvv_labels]
    dct["add"] = [add_data, add_labels]
    dct["b"] = [b_data, b_labels]

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """

    datasets = {
        "nc": (nc_data,nc_labels),
        "nm": (nm_data,nm_labels),
        "bvv": (bvv_data,bvv_labels),
        "add": (add_data,add_labels),
        "b": (b_data,b_labels)
        }

    num_clusters = [2]
    dataset_keys = ['nc', 'nm', 'bvv', 'add', 'b']
    linkage_types = ['single', 'complete', 'ward', 'average']
    pdf_filename = "report_4B.pdf"
    pdf_pages = []


    fig, axes = plt.subplots(len(linkage_types), len(dataset_keys), figsize=(20, 16))
    fig.suptitle('Scatter plots for different datasets and linkage types (2 clusters)', fontsize=16)
            
    for i, linkage_type in enumerate(linkage_types):
        for j, dataset_key in enumerate(dataset_keys):
            data, labels = datasets[dataset_key]

            for k in num_clusters:
                predicted_labels = fit_hierarchical_cluster_linkage(given_datasets[dataset_key], n_clusters=k, linkage=linkage_type)

                ax = axes[i, j]
                ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
                ax.set_title(f'{linkage_type.capitalize()} Linkage\n{dataset_key}, k={k}')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_pages.append(fig)
    plt.close(fig)

    with PdfPages(pdf_filename) as pdf:
        for page in pdf_pages:
            pdf.savefig(page)

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    # dct is the function described above in 4.C
    dct = answers["4A: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
