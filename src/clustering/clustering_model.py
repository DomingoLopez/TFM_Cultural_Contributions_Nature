from datetime import datetime
import os
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import pylab as plt
from loguru import logger
import sys
from abc import ABC, abstractmethod

from typing import Optional, Tuple
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


class ClusteringModel(ABC):
    """
    Base abstract class for clustering models.

    This class provides a template for implementing clustering models with methods
    for running the clustering, saving clustering results and plots, and performing
    basic PCA for 2D representation of clusters.
    """

    def __init__(self, 
                 data: pd.DataFrame,
                 model_name: str):
        """
        Initialize the clustering model with data.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset on which clustering will be performed.
        """
        self.data = data
        self.model_name = model_name
        # Setting up directories for saving results based on model_name
        # Theese are base directories. Every model will define their results
        # directory tree
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        base_path = Path(__file__).resolve().parent.parent / "results" / model_name / timestamp
        self.folder_run = base_path
        self.folder_plots = base_path / "plots"
        self.folder_results = base_path / "results" 
        
        # Create folders if they don't exist
        os.makedirs(self.folder_plots, exist_ok=True)
        os.makedirs(self.folder_results, exist_ok=True)

    @abstractmethod
    def run(self):
        """
        Run the clustering experiment.

        This method should be implemented by subclasses to perform specific clustering
        operations on the dataset provided at initialization. It defines the main 
        execution flow of the clustering.
        """
        pass

    def save_clustering_plot(
        self,
        X: pd.DataFrame, 
        c: Optional[np.ndarray] = None, 
        centroids: Optional[np.ndarray] = None,
        i: int = 0, 
        j: int = 0, 
        figs: Tuple[int, int] = (9, 7),
        save_path: str = ""
    ):
        """
        Plots a 2D representation of the dataset and its associated clusters.

        This method saves a plot showing the clustering of the 2D-reduced data points,
        optionally marking the cluster centroids if provided.

        Parameters
        ----------
        X : pd.DataFrame
            Data points to plot, with each row representing a sample in 2D space.
        c : Optional[np.ndarray]
            Cluster labels for each point.
        centroids : Optional[np.ndarray]
            Coordinates of cluster centroids in 2D space.
        i : int
            Index of the feature for the x-axis.
        j : int
            Index of the feature for the y-axis.
        figs : Tuple[int, int]
            Size of the figure in inches.
        save_path : str
            Path to store the plot image.
        """

        # color mapping for clusters
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#FFFF00', '#0000FF','#FF9D0A','#00B6FF','#F200FF','#FF6100'])
        # Plotting frame
        plt.figure(figsize=figs)
        # Plotting points with seaborn
        sns.scatterplot(x=X.iloc[:, i], y=X.iloc[:, j], hue=c, palette=cmap_bold.colors, s=30)
        # Plotting centroids
        if centroids is not None:
            sns.scatterplot(x=centroids[:, i], y=centroids[:, j], marker='D',palette=cmap_bold.colors, hue=range(centroids.shape[0]), s=100,edgecolors='black')
        # Save plot making sure dir exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')


    def save_clustering_result(
        self,
        k: list,
        score_silhouette: list,
        score_davies: list,
        error: list,
        file_path_result_plot: str,
        file_path_result_csv: str,
        file_path_result_error: str
    ):
        """
        Saves clustering scores and generates a plot of the scores.

        This method generates a line plot showing the silhouette and Davies-Bouldin 
        scores for different cluster configurations, and saves both the plot and the 
        raw scores in a CSV file.

        Parameters
        ----------
        k : list
            List of cluster counts or other variable values tested during clustering.
        score_silhouette : list
            List of silhouette scores for each clustering configuration.
        score_davies : list
            List of Davies-Bouldin scores for each clustering configuration.
        error: list
            Error inertia from clustering
        file_path_result_plot: str
            Path for saving plot 
        file_path_result_csv: str
            Path for saving csv with results
        file_path_result_error: str
            Path for saving error inertia just in case
        """
                
        # Save making sure dir exists
        os.makedirs(os.path.dirname(file_path_result_plot), exist_ok=True)
        os.makedirs(os.path.dirname(file_path_result_csv), exist_ok=True)
        if len(error) > 1:
            os.makedirs(os.path.dirname(file_path_result_error), exist_ok=True)
        
        # Plot scores
        if len(score_silhouette) > 1:
            scores = pd.DataFrame({'k_value':k,'sil_score': score_silhouette,'davies_score': score_davies})
            # plot
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=scores, x='k_value', y='sil_score', label='Silhouette Score')
            sns.lineplot(data=scores, x='k_value', y='davies_score', label='Davies Score')
            plt.title('Scores en función de k_value')
            plt.xlabel('k_value')
            plt.ylabel('Scores')
            plt.legend()
            # save figure
            plt.savefig(file_path_result_plot, bbox_inches='tight')
            # save csv
            scores.to_csv(file_path_result_csv, index=False)
        
        if len(error) > 1:
            # Plot error inertia
            errors = pd.DataFrame({'k_value':k,'error_inertia': error})
            # plot
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=errors, x='k_value', y='error_inertia', label='Error inertia')
            plt.title('Errors vs k')
            plt.xlabel('k_value')
            plt.ylabel('error inertia')
            plt.legend()
            # save figure
            plt.savefig(file_path_result_error, bbox_inches='tight')



    def find_and_save_clustering_knn_points(self, n_neighbors, metric, cluster_centers, save_path ):
        """
        Finds the k-nearest neighbors for each centroid of clusters and saves the results in a CSV file.

        This method calculates the `n_neighbors` closest points in the dataset to each centroid in 
        `cluster_centers` using the specified `metric`. It then saves the results in a CSV file where
        each column corresponds to a cluster (identified by its centroid), and each row lists the indices 
        of the closest neighbors to that centroid.

        Parameters
        ----------
        n_neighbors : int
            Number of nearest neighbors to find for each centroid.

        metric : str
            Distance metric to use for finding neighbors (e.g., 'euclidean', 'manhattan').

        cluster_centers : array-like of shape (n_clusters, n_features)
            Coordinates of cluster centroids for which the nearest neighbors are computed.

        save_path : str
            File path where the CSV file containing nearest neighbors information will be saved.

        Attributes
        ----------
        closest_neighbors : dict
            Dictionary where each key is the cluster index and the value is an array of indices representing
            the nearest neighbors for that cluster's centroid.

        Side Effects
        ------------
        - Creates the directory specified by `save_path` if it does not already exist.
        - Saves a CSV file containing the indices of the nearest neighbors for each cluster's centroid.

        Example
        -------
        >>> self.find_and_save_clustering_knn_points(3, 'euclidean', cluster_centers, 'output/knn_points.csv')

        """
        closest_neighbors = {}
        used_metric = metric if metric in ('cityblock', 'cosine','euclidean','haversine','l1','l2','manhattan','nan_euclidean') else 'euclidean'
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=used_metric, algorithm='auto').fit(self.data)
        for idx, centroid in enumerate(cluster_centers):
            distances, indices = nbrs.kneighbors([centroid])
            if idx not in closest_neighbors:
                closest_neighbors[idx] = []
            closest_neighbors[idx]= indices.flatten()
            
        # check dir exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Transform to df and save to csv
        knn_data = {f"Cluster_{idx}": neighbors for idx, neighbors in closest_neighbors.items()}
        df_closest_neighbors = pd.DataFrame(knn_data)
        df_closest_neighbors.to_csv(save_path, index=False)
                                  

    def do_PCA_for_representation(self, df, centers):
        """
        Performs PCA to reduce data and cluster centers to 2D for plotting.

        This function applies PCA (Principal Component Analysis) to the data and 
        optionally to the cluster centers, reducing them to 2D space for visualization 
        purposes.

        Parameters
        ----------
        df : pd.DataFrame
            Data points to reduce, where each row represents a sample.
        centers : np.ndarray
            Coordinates of cluster centroids before dimensionality reduction.

        Returns
        -------
        pca_df : pd.DataFrame
            Data points reduced to 2D space.
        pca_centers : np.ndarray
            Cluster centroids reduced to 2D space.
        """
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df.values)
        pca_df = pd.DataFrame(data=pca_result)
        pca_centers = pca.transform(centers)

        return pca_df, pca_centers
