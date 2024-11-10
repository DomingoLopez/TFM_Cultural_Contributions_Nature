from collections import Counter
import os
from pathlib import Path
import pickle
import sys
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
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
                 images: list,
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
        # clustering
        # optimization
        # scaler
        # dim_reduction
        # dimensions
        # embeddings
        # n_clusters
        # best_params
        # centers
        # labels
        # label_counter
        # noise_not_noise
        # silhouette_noise_ratio
        # penalty
        # penalty_range
        # best_value_w_penalty
        # best_value_w/o_penalty

        self.trial_result = trial_result
        self.images = images
        for k,v in trial_result.items():
            setattr(self, k, v)
        self._cache = cache
        self._verbose = verbose

        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")



    def get_experiment_name(self):

        return f"{self.clustering}_{self.optimization}_{self.dimensions}_dims_{self.dim_reduction}_{round(self.trial_result.get('best_value_w/o_penalty') * 100):03d}"



    def get_cluster_images_dict(self, knn=None):
        """
        Finds the k-nearest neighbors for each centroid of clusters among points that belong to the same cluster.
        Returns knn points for each cluster in dict format in case knn is not None

        Parameters
        ----------
        knn : int
            Number of nearest neighbors to find for each centroid

        Returns
        -------
        sorted_cluster_images_dict : dictionary with images per cluster (as key)
        """

        cluster_images_dict = {}
        labels = self.labels

        if knn is not None:
            # used_metric = (
            #     self.best_params.get("metric")
            #     if self.best_params.get("metric") in ('cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean')
            #     else 'euclidean'
            # )
            used_metric = "euclidean"
            
            for idx, centroid in enumerate(self.centers):
                # Filter points based on label mask over embeddings
                cluster_points = self.embeddings.values[labels == idx]
                cluster_images = [self.images[i] for i in range(len(self.images)) if labels[i] == idx]
                # Adjust neighbors, just in case
                n_neighbors_cluster = min(knn, len(cluster_points))
                
                nbrs = NearestNeighbors(n_neighbors=n_neighbors_cluster, metric=used_metric, algorithm='auto').fit(cluster_points)
                distances, indices = nbrs.kneighbors([centroid])
                closest_indices = indices.flatten()
                
                # Get images for each cluster
                cluster_images_dict[idx] = [cluster_images[i] for i in closest_indices]

            # Get noise (-1)
            cluster_images_dict[-1] = [self.images[i] for i in range(len(self.images)) if labels[i] == -1]
            
        else:
            for i, label in enumerate(labels):
                if label not in cluster_images_dict:
                    cluster_images_dict[label] = []
                cluster_images_dict[label].append(self.images[i])
        
        # Sort dictionary
        sorted_cluster_images_dict = dict(sorted(cluster_images_dict.items()))
        return sorted_cluster_images_dict
    
    
    
    
    
    
    # def find_clustering_cosine_similarity_points(self, n_neighbors, cluster_centers, labels):
    #     """
    #     Finds the k-nearest neighbors for each centroid of clusters within the same cluster, using cosine similarity.

    #     This method calculates the `n_neighbors` closest points in the dataset to each centroid in 
    #     `cluster_centers` using cosine similarity, considering only points in the same cluster.

    #     Parameters
    #     ----------
    #     n_neighbors : int
    #         Number of nearest neighbors to find for each centroid.

    #     cluster_centers : array-like of shape (n_clusters, n_features)
    #         Coordinates of cluster centroids for which the nearest neighbors are computed.

    #     labels : array-like of shape (n_samples,)
    #         Cluster labels for each data point in the dataset.

    #     Returns
    #     -------
    #     df_closest_neighbors : pandas.DataFrame
    #         DataFrame where each column corresponds to a cluster index, and rows contain indices
    #         of the closest neighbors within that cluster based on cosine similarity for each cluster's centroid.
    #     """
    #     closest_neighbors = {}

    #     for idx, centroid in enumerate(cluster_centers):
    #         # Filtrar puntos que pertenecen al mismo cluster
    #         cluster_points = self.data.values[labels == idx]
            
    #         # Asegurarse de que haya suficientes puntos en el cluster para el número de vecinos solicitado
    #         if len(cluster_points) < n_neighbors:
    #             n_neighbors_cluster = len(cluster_points)
    #         else:
    #             n_neighbors_cluster = n_neighbors
            
    #         # Calcular similitud de coseno entre el centroide y los puntos del cluster
    #         similarities = cosine_similarity(cluster_points, centroid.reshape(1, -1)).flatten()
            
    #         # Obtener los índices de los puntos más similares en el cluster
    #         top_indices = np.argsort(similarities)[-n_neighbors_cluster:]  # Obtener índices de los más altos
    #         closest_neighbors[idx] = top_indices[::-1]  # Ordenar de más a menos similar

    #     # Convertir a DataFrame
    #     knn_data = {f"Cluster_{idx}": neighbors for idx, neighbors in closest_neighbors.items()}
    #     df_closest_neighbors = pd.DataFrame(knn_data)

    #     return df_closest_neighbors
    

if __name__ == "__main__":
    pass
