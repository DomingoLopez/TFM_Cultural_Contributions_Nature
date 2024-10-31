import os
from pathlib import Path
import sys
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_model import ClusteringModel


class KMeansClustering(ClusteringModel):
    """
    KMeans clustering model class inheriting from ClusteringModel.

    This class implements the KMeans clustering algorithm on a dataset and 
    provides methods to run clustering, calculate metrics, and save results 
    including plots and score data.
    """

    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Initialize the KMeansClustering model.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be clustered.
        **kwargs : dict
            Additional parameters for customization or future expansion.
        """
        super().__init__(data, model_name="kmeans")
        
        
    def run_optuna(self, evaluation_method: str = "silhouette", n_trials: int = 50,penalty: str="linear", penalty_range: tuple =(2,8)):
        """
        Run Optuna optimization for the KMeans clustering model with a specified evaluation method.

        This method performs an Optuna hyperparameter optimization to tune the KMeans clustering 
        algorithm. It defines a model builder function to set the hyperparameter ranges specific to 
        KMeans, including the number of clusters (`n_clusters`), initialization method (`init`), 
        number of centroid seeds (`n_init`), and maximum number of iterations (`max_iter`). 
        The optimization goal is to maximize the silhouette score or minimize the Davies-Bouldin score 
        based on the specified `evaluation_method`.

        Parameters
        ----------
        evaluation_method : str, optional
            The evaluation metric to optimize. Can be either 'silhouette' for maximizing the 
            silhouette score, or 'davies_bouldin' for minimizing the Davies-Bouldin score. 
            Defaults to 'silhouette'.
        n_trials : int, optional
            The number of trials to run in Optuna optimization. Defaults to 50.

        Returns
        -------
        optuna.study.Study
            The Optuna study object containing details of the optimization process, including 
            the best hyperparameters found and the associated evaluation score.

        Notes
        -----
        - The method defines a `model_builder` function that constructs a KMeans model with 
        hyperparameters suggested by each Optuna trial.
        - Calls the `run_optuna_generic` method from the base class `ClusteringModel`, which 
        manages the Optuna optimization process and progress tracking.
        - The optimization direction is set based on the specified evaluation metric.

        Example
        -------
        >>> kmeans_clustering = KMeansClustering(data)
        >>> study = kmeans_clustering.run_optuna(evaluation_method="davies_bouldin", n_trials=100)
        >>> print("Best parameters:", study.best_params)
        >>> print("Best score:", study.best_value)
        """
        
        # Define the model builder function for KMeans
        def model_builder(trial):
            return KMeans(
                n_clusters=trial.suggest_int('n_clusters', 2, 10),
                init=trial.suggest_categorical('init', ['k-means++', 'random']),
                n_init=trial.suggest_int('n_init', 10, 20),
                max_iter=trial.suggest_int('max_iter', 100, 300)
            )

        # Call the generic Optuna optimization method
        return self.run_optuna_generic(model_builder, evaluation_method, n_trials,penalty, penalty_range)
        
        

    def run_basic_experiment(self):
        """
        Execute the KMeans clustering process on the dataset.

        This method performs KMeans clustering across a range of cluster values (k) 
        from 2 to 9. For each value of k, it calculates the silhouette and Davies-Bouldin 
        scores, saves plots for each clustering configuration, and stores the results.

        Attributes
        ----------
        error : list
            List to store the inertia (sum of squared distances to closest cluster center) 
            for each k.
        k_ : list
            List of cluster values tested.
        silhouette_coefficients : list
            List of silhouette scores for each k.
        davies_bouldin_coefficients : list
            List of Davies-Bouldin scores for each k.
        
        Side Effects
        ------------
        - Saves clustering plots for each configuration in `folder_plots`.
        - Saves score plots and CSV files in `folder_results`.
        """
        
        # Collect indexes
        error = []
        k_ = []
        silhouette_coefficients = []
        davies_bouldin_coefficients = []

        # TODO: We could try different metrics too
        # Perform clustering for each k
        for k in range(2, 9):
            # Define params
            params = {
                "n_clusters": k,
                "n_init": 100,
                "init": "k-means++",
                "random_state": 1234
            }
            # Run KMeans
            kmeans = KMeans(**params).fit(self.data)
            error.append(kmeans.inertia_)
            
            # Find the 3 nearest neighbors for each centroid and save it in file
            super().find_and_save_clustering_knn_points(
                n_neighbors=3, 
                metric=params.get("metric", "euclidean"),
                cluster_centers=kmeans.cluster_centers_,
                save_path = os.path.join(self.folder_plots, f"n_clusters_{k}/knn_points/points.csv")
            )
                
            # CALCULATE RESULT DIRS (PLOTTING + KNN CENTROIDS)
            file_path_plot = os.path.join(self.folder_plots, f"n_clusters_{k}/plot.png")
            
            # REPRESENTATION
            pca_df, pca_centers = super().do_PCA_for_representation(self.data, kmeans.cluster_centers_)
            super().save_clustering_plot(pca_df, kmeans.labels_, pca_centers, i=0, j=1, save_path=file_path_plot)
            

            # SCORERS
            score_silhouette = silhouette_score(self.data, kmeans.labels_)
            score_davies = davies_bouldin_score(self.data, kmeans.labels_)
            silhouette_coefficients.append(score_silhouette)
            davies_bouldin_coefficients.append(score_davies)
            k_.append(k)


        # CALCULATE RESULT DIRS (RESULTS)
        file_path_result_csv = os.path.join(self.folder_results, f"result.csv")
        file_path_result_plot = os.path.join(self.folder_results, f"result.png")
        file_path_result_error = os.path.join(self.folder_results, f"errors.png")
        

        # Save the scores and generate plots
        super().save_clustering_result(
            k_, 
            silhouette_coefficients, 
            davies_bouldin_coefficients, 
            error,
            # paths
            file_path_result_plot,
            file_path_result_csv,
            file_path_result_error
        )


if __name__ == "__main__":
    # Test the KMeansClustering class with a sample dataset
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    data = pd.DataFrame(data, columns=['x', 'y'])
    kmeans_clustering = KMeansClustering(data)
    kmeans_clustering.run()
    print("KMeans clustering complete. Results and plots saved.")
