import optuna
from datetime import datetime
import os
import pickle
from pathlib import Path
from tqdm import tqdm
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
from sklearn.metrics.pairwise import cosine_similarity


class ClusteringModel(ABC):
    """
    Base abstract class for clustering models.

    This class provides a template for implementing clustering models with methods
    for running the clustering, saving clustering results and plots, and performing
    basic PCA for 2D representation of clusters.
    """

    def __init__(self, 
                 data: pd.DataFrame,
                 model_name: str,
                 params: Optional[dict] = {}):
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
    def run_basic_experiment(self):
        """
        Run the clustering experiment for some default params given in each implementation.
        This method is just to run some experiments in order to see the problem from different
        points of view and as a way to improve incoming  implementations.
        """
        pass
    
    
    def get_cluster_centers(self, labels):
        """
        Calculate the centers of clusters given data points and their cluster labels.

        This function computes the center of each cluster by calculating the mean 
        position of all points within each cluster. It is suitable for use with 
        clustering algorithms where direct access to cluster centers is unavailable, 
        such as when using `fit_predict` in KMeans or AgglomerativeClustering.

        Parameters
        ----------
        data : array-like or pandas.DataFrame of shape (n_samples, n_features)
            The dataset where each row represents a data point with multiple features.
        labels : array-like of shape (n_samples,)
            Cluster labels assigned to each data point, with each unique label 
            representing a separate cluster. Points labeled as -1 are typically 
            considered noise and can be excluded if desired.

        Returns
        -------
        centers : numpy.ndarray of shape (n_clusters, n_features)
            An array containing the calculated center of each cluster. Each row 
            corresponds to the center of one cluster, with the same number of 
            features as the input data.

        Example
        -------
        >>> from sklearn.cluster import KMeans
        >>> import numpy as np
        >>> data = np.random.rand(100, 2)
        >>> kmeans = KMeans(n_clusters=3)
        >>> labels = kmeans.fit_predict(data)
        >>> centers = get_cluster_centers(data, labels)
        >>> print(centers)

        Notes
        -----
        - This method calculates the mean of each cluster's points to find a center, 
        which approximates the centroid.
        - For algorithms like KMeans, which explicitly compute centroids, the centers 
        obtained here should closely match the model's centroids.
        """
        unique_labels = np.unique(labels)
        centers = []
        for label in unique_labels:
            if label != -1:  # Ignore noise
                cluster_points = self.data.values[labels == label]
                cluster_center = np.mean(cluster_points, axis=0)
                centers.append(cluster_center)

        if centers:
            centers = np.array(centers)
        
        return centers
            
            
    
    def run_optuna_generic(self, model_builder, evaluation_method="silhouette", n_trials=50, penalty="linear", penalty_range=(2,8)):
        """
        Generic Optuna optimization for clustering models with configurable penalty types.

        Parameters
        ----------
        model_builder : Callable[[optuna.trial.Trial], clustering_model]
            Function that builds the clustering model with hyperparameters suggested by the Optuna trial.
        evaluation_method : str
            The evaluation metric to optimize ('silhouette' or 'davies_bouldin').
        n_trials : int
            The number of trials for Optuna. Default is 50.
        penalty : str, optional
            The type of penalty to apply based on `n_clusters`. Options are:
            - "linear": Linear penalty based on `n_clusters`.
            - "proportional": Proportional penalty inversely related to `n_clusters`.
            - "range": Proportional penalty only when `n_clusters` is outside a specified range.
        """
        
        # Define acceptable range for "range" penalty type
        if penalty_range is not None and penalty == "range":
            min_clusters, max_clusters = penalty_range
        
        # Objective function
        def objective(trial):
            # Build the model with suggested hyperparameters
            model = model_builder(trial)
            
            # Fit and predict
            labels = model.fit_predict(self.data)
            
            # Get number of clusters (excluding noise)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            trial.set_user_attr("n_clusters", n_clusters)
            
            # Calculate cluster centers
            centers = self.get_cluster_centers(labels)
            trial.set_user_attr("centers", centers)  # Store centers in trial attributes

            
            # Evaluate model
            if n_clusters > 1:
                # Calculate the original score without penalty
                if evaluation_method == "silhouette":
                    score_original = silhouette_score(self.data[labels != -1], labels[labels != -1])
                elif evaluation_method == "davies_bouldin":
                    score_original = davies_bouldin_score(self.data[labels != -1], labels[labels != -1])
                else:
                    raise ValueError("Evaluation method not supported. Use 'silhouette' or 'davies_bouldin' instead.")

                # Apply the selected penalty type
                if penalty == "linear":
                    # Linear penalty: subtract or add 0.1 * n_clusters
                    adjustment = 0.1 * n_clusters
                    score_penalized = score_original - adjustment if evaluation_method == "silhouette" else score_original + adjustment

                elif penalty == "proportional":
                    # Proportional penalty: multiply by a factor based on n_clusters
                    penalty_factor = 1 - (1 / n_clusters) if evaluation_method == "silhouette" else 1 + (1 / n_clusters)
                    score_penalized = score_original * penalty_factor

                elif penalty == "range":
                    # Range-based penalty: penalize if n_clusters is outside [min_clusters, max_clusters]
                    if n_clusters < min_clusters:
                        adjustment = 0.1 * (min_clusters - n_clusters)
                        score_penalized = score_original - adjustment if evaluation_method == "silhouette" else score_original + adjustment
                    elif n_clusters > max_clusters:
                        adjustment = 0.1 * (n_clusters - max_clusters)
                        score_penalized = score_original - adjustment if evaluation_method == "silhouette" else score_original + adjustment
                    else:
                        score_penalized = score_original  # No penalty if within range

                else:
                    raise ValueError("Penalty type not supported. Use 'linear', 'proportional', or 'range'.")
                    
            else:
                # Penalize single/no cluster cases
                score_original = -1 if evaluation_method == "silhouette" else float('inf')
                score_penalized = score_original

            # Log the original score for later reference
            trial.set_user_attr("score_original", score_original)
            
            return score_penalized

        # Set optimization direction
        direction = "maximize" if evaluation_method == "silhouette" else "minimize"
        
        # Execute Optuna optimization with tqdm
        pbar = tqdm(total=n_trials, desc="Optuna Optimization")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda study, trial: pbar.update(1)])
        pbar.close()

        return study

    

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
        colors = ['#FF0000', '#00FF00', '#FFFF00', '#0000FF', '#FF9D0A', '#00B6FF', '#F200FF', '#FF6100']
        cmap_bold = ListedColormap(colors)
        # Plotting frame
        plt.figure(figsize=figs)
        # Plotting points with seaborn
        sns.scatterplot(x=X.iloc[:, i], y=X.iloc[:, j], hue=c, palette=cmap_bold.colors, s=30, hue_order=sorted(set(c)))  # Ensures that -1 appears first in the legend if present)
        # Plotting centroids
        if centroids is not None:
            sns.scatterplot(x=centroids[:, i], y=centroids[:, j], marker='D',palette=colors[1:] if -1 in set(c) else colors[:], hue=range(centroids.shape[0]), s=100,edgecolors='black')
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



    def find_clustering_knn_points(self, n_neighbors, metric, cluster_centers):
        """
        Finds the k-nearest neighbors for each centroid of clusters. Returns knn points for
        each cluster in dataframe style.

        This method calculates the `n_neighbors` closest points in the dataset to each centroid in 
        `cluster_centers` using the specified `metric`. 

        Parameters
        ----------
        n_neighbors : int
            Number of nearest neighbors to find for each centroid.

        metric : str
            Distance metric to use for finding neighbors (e.g., 'euclidean', 'manhattan').

        cluster_centers : array-like of shape (n_clusters, n_features)
            Coordinates of cluster centroids for which the nearest neighbors are computed.

        Attributes
        ----------
        closest_neighbors : dict
            Dictionary where each key is the cluster index and the value is an array of indices representing
            the nearest neighbors for that cluster's centroid.

        Example
        -------
        >>> self.find_clustering_knn_points(3, 'euclidean', cluster_centers, 'output/knn_points.csv')

        """
        closest_neighbors = {}
        used_metric = metric if metric in ('cityblock', 'cosine','euclidean','haversine','l1','l2','manhattan','nan_euclidean') else 'euclidean'
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=used_metric, algorithm='auto').fit(self.data)
        for idx, centroid in enumerate(cluster_centers):
            distances, indices = nbrs.kneighbors([centroid])
            if idx not in closest_neighbors:
                closest_neighbors[idx] = []
            closest_neighbors[idx]= indices.flatten()
            
        # Transform to df and save to csv
        knn_data = {f"Cluster_{idx}": neighbors for idx, neighbors in closest_neighbors.items()}
        df_closest_neighbors = pd.DataFrame(knn_data)

        return df_closest_neighbors
    
    
    
    
    def find_clustering_cosine_similarity_points(self, n_neighbors, cluster_centers):
        """
        Finds the k-nearest neighbors for each centroid of clusters using cosine similarity.

        This method calculates the `n_neighbors` closest points in the dataset to each centroid in 
        `cluster_centers` using cosine similarity.

        Parameters
        ----------
        n_neighbors : int
            Number of nearest neighbors to find for each centroid.

        cluster_centers : array-like of shape (n_clusters, n_features)
            Coordinates of cluster centroids for which the nearest neighbors are computed.

        Returns
        -------
        df_closest_neighbors : pandas.DataFrame
            DataFrame where each column corresponds to a cluster index, and rows contain indices
            of the closest neighbors based on cosine similarity for that cluster's centroid.
        """
        closest_neighbors = {}
        for idx, centroid in enumerate(cluster_centers):
            similarities = cosine_similarity(self.data, centroid.reshape(1, -1)).flatten()
            # Get index of n neighbors with max similarity
            top_indices = np.argsort(similarities)[-n_neighbors:]  # Get highest
            closest_neighbors[idx] = top_indices[::-1]  # Invert to order from more to less similarity
        
        # Converto to df
        knn_data = {f"Cluster_{idx}": neighbors for idx, neighbors in closest_neighbors.items()}
        df_closest_neighbors = pd.DataFrame(knn_data)

        return df_closest_neighbors
    
    


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
