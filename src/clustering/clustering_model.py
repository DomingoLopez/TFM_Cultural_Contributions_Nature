import datetime
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


class ClusteringModel(ABC):
    """
    Base abstract class for clustering models.

    This class provides a template for implementing clustering models with methods
    for running the clustering, saving clustering results and plots, and performing
    basic PCA for 2D representation of clusters.
    """

    def __init__(self, 
                 data: pd.DataFrame):
        """
        Initialize the clustering model with data.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset on which clustering will be performed.
        """
        self.data = data

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
        save_path: str = "",
        save_path_csv: str = "",
        save_path_error: str =""
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
        save_path : str
            Path to save the plot with scores.
        save_path_csv : str
            Path to save the CSV file containing scores data.
        save_path_error : str
            Path to save the plot with error inertia 
        """
        
         # Save making sure dir exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
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
            plt.savefig(save_path, bbox_inches='tight')
            # save csv
            scores.to_csv(save_path_csv, index=False)
        
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
            plt.savefig(save_path_error, bbox_inches='tight')



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
