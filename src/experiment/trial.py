from collections import Counter
import os
from pathlib import Path
import pickle
import sys
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_factory import ClusteringFactory
from src.clustering.clustering_model import ClusteringModel
from src.eda.eda import EDA


class Trial():
    """
    Single trial from experiment.
    It has de same results as a row of a experiment result object. 
    We can do things with trial, for example operate with labels, noise of this trial,
    results, etc.
    """
    def __init__(self, 
                 trial_result: dict,
                 cache:bool= True, 
                 verbose:bool= False,
                 **kwargs):
        """
        Loads a trial in memory to use it, manipulate it and use as input for Llava-1.5 multimodal.
        Args:
            trial_result (dict): trial of some experiment
            cache (bool): If True, caching is enabled.
            verbose (bool): If True, enables verbose logging.
            **kwargs: Additional keyword arguments.
        """
        # Setup attrs
        self.trial_result = trial_result
        for k,v in trial_result.items():
            setattr(self, k, v)
        self._cache = cache
        self._verbose = verbose

        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")




    def find_clustering_knn_points(self, n_neighbors):
        """
        Finds the k-nearest neighbors for each centroid of clusters among points that belong to the same cluster.
        Returns knn points for each cluster in dataframe format.

        Parameters
        ----------
        n_neighbors : int
            Number of nearest neighbors to find for each centroid

        Returns
        -------
        df_closest_neighbors : pandas.DataFrame
            DataFrame where each column corresponds to a cluster index, and rows contain indices
            of the closest neighbors within that cluster for the centroid.
        """
        # get trial mandatory params
        
        
        
        closest_neighbors = {}
        
        
        
        used_metric = metric if metric in ('cityblock', 'cosine','euclidean','haversine','l1','l2','manhattan','nan_euclidean') else 'euclidean'
        
        for idx, centroid in enumerate(cluster_centers):
            # Filter poitns that belongs to cluster. If not
            # we could end up with points from other clusters
            cluster_points = self.data.values[labels == idx]
            
            # Make sure there are more cluster points that neighbors required
            if len(cluster_points) < n_neighbors:
                n_neighbors_cluster = len(cluster_points)
            else:
                n_neighbors_cluster = n_neighbors
            
            # Do NNeighbors
            nbrs = NearestNeighbors(n_neighbors=n_neighbors_cluster, metric=used_metric, algorithm='auto').fit(cluster_points)
            distances, indices = nbrs.kneighbors([centroid])
            
            closest_neighbors[idx] = indices.flatten()  # Almacenar los índices locales del cluster

        # Transform to df
        knn_data = {f"Cluster_{idx}": neighbors for idx, neighbors in closest_neighbors.items()}
        df_closest_neighbors = pd.DataFrame(knn_data)

        return df_closest_neighbors
    
    
    
    
    def find_clustering_cosine_similarity_points(self, n_neighbors, cluster_centers, labels):
        """
        Finds the k-nearest neighbors for each centroid of clusters within the same cluster, using cosine similarity.

        This method calculates the `n_neighbors` closest points in the dataset to each centroid in 
        `cluster_centers` using cosine similarity, considering only points in the same cluster.

        Parameters
        ----------
        n_neighbors : int
            Number of nearest neighbors to find for each centroid.

        cluster_centers : array-like of shape (n_clusters, n_features)
            Coordinates of cluster centroids for which the nearest neighbors are computed.

        labels : array-like of shape (n_samples,)
            Cluster labels for each data point in the dataset.

        Returns
        -------
        df_closest_neighbors : pandas.DataFrame
            DataFrame where each column corresponds to a cluster index, and rows contain indices
            of the closest neighbors within that cluster based on cosine similarity for each cluster's centroid.
        """
        closest_neighbors = {}

        for idx, centroid in enumerate(cluster_centers):
            # Filtrar puntos que pertenecen al mismo cluster
            cluster_points = self.data.values[labels == idx]
            
            # Asegurarse de que haya suficientes puntos en el cluster para el número de vecinos solicitado
            if len(cluster_points) < n_neighbors:
                n_neighbors_cluster = len(cluster_points)
            else:
                n_neighbors_cluster = n_neighbors
            
            # Calcular similitud de coseno entre el centroide y los puntos del cluster
            similarities = cosine_similarity(cluster_points, centroid.reshape(1, -1)).flatten()
            
            # Obtener los índices de los puntos más similares en el cluster
            top_indices = np.argsort(similarities)[-n_neighbors_cluster:]  # Obtener índices de los más altos
            closest_neighbors[idx] = top_indices[::-1]  # Ordenar de más a menos similar

        # Convertir a DataFrame
        knn_data = {f"Cluster_{idx}": neighbors for idx, neighbors in closest_neighbors.items()}
        df_closest_neighbors = pd.DataFrame(knn_data)

        return df_closest_neighbors
    

    



if __name__ == "__main__":
    pass
