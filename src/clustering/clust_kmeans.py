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
        
        
    def run_optuna(self, evaluation_method: str = "silhouette", n_trials: int = 50):
        """
        Run Optuna optimization for KMeans with a specified evaluation method.

        Parameters
        ----------
        evaluation_method : str
            The evaluation metric to optimize ('silhouette' or 'davies_bouldin').
        n_trials : int
            The number of trials for Optuna. Default is 50.
        """

        # Define the objective function for Optuna
        def objective(trial):
            # Suggest hyperparameters
            n_clusters = trial.suggest_int('n_clusters', 2, 10)
            init = trial.suggest_categorical('init', ['k-means++', 'random'])
            n_init = trial.suggest_int('n_init', 10, 20)
            max_iter = trial.suggest_int('max_iter', 100, 300)

            # Create KMeans model with suggested hyperparameters
            model = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                random_state=1234
            )

            # Train and predict labels
            labels = model.fit_predict(self.data)

            # Evaluate model
            if len(set(labels)) > 1:
                if evaluation_method == "silhouette":
                    score = silhouette_score(self.data, labels)
                elif evaluation_method == "davies_bouldin":
                    score = -davies_bouldin_score(self.data, labels)  # Negative, as DB index is better when lower
                else:
                    raise ValueError("Método de evaluación no soportado. Usa 'silhouette' o 'davies_bouldin'.")
            else:
                score = -1  # Penalty in case of obtaining no clusters

            return score

        # Call the parent method to run Optuna with the defined objective function
        study = super().optimize_with_optuna(objective, n_trials=n_trials, direction="maximize")
        return study
        
        

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
