from datetime import datetime
#import hdbscan
import optuna
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


class HDBSCANClustering(ClusteringModel):
    """
    HDBSCAN clustering model class inheriting from ClusteringModel.

    This class implements the HDBSCAN clustering algorithm on a dataset and 
    provides methods to run clustering, calculate metrics, and save results 
    including plots and score data.
    """

    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Initialize the HDBSCANClustering model.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be clustered.
        **kwargs : dict
            Additional parameters for customization or future expansion.
        """
        super().__init__(data, model_name="hdbscan")
        
        
    def run_optuna(self, evaluation_method: str = "silhouette", n_trials: int = 50):
        """
        Run Optuna optimization for HDBSCAN with a specified evaluation method.

        Parameters
        ----------
        evaluation_method : str
            The evaluation metric to optimize ('silhouette' or 'davies_bouldin').
        n_trials : int
            The number of trials for Optuna. Default is 50.
        """

        # Objetive function for optuna
        def objective(trial):
            # Hyperparams suggestions
            min_cluster_size = trial.suggest_int('min_cluster_size', 50, 150)
            min_samples = trial.suggest_int('min_samples', 25, 75)
            cluster_selection_epsilon = trial.suggest_loguniform('cluster_selection_epsilon', 0.01, 1.0)
            alpha = trial.suggest_float('alpha', 0.5, 2.0)
            metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev'])
            cluster_selection_method = trial.suggest_categorical('cluster_selection_method', ['eom', 'leaf'])
            gen_min_span_tree = trial.suggest_categorical('gen_min_span_tree', [True, False])

            # Create model
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                alpha=alpha,
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                gen_min_span_tree=gen_min_span_tree
            )

            # Train and predict labels
            labels = model.fit_predict(self.data)

            # Eval model
            if len(set(labels)) > 1:  
                if evaluation_method == "silhouette":
                    score = silhouette_score(self.data[labels != -1], labels[labels != -1])
                elif evaluation_method == "davies_bouldin":
                    score = -davies_bouldin_score(self.data[labels != -1], labels[labels != -1])  # Negative, cause dbindex better when lower
                else:
                    raise ValueError("Método de evaluación no soportado. Usa 'silhouette' o 'davies_bouldin'.")
            else:
                score = -1  # Penalty in case of obtaining no clusters or noise

            return score

        # Llamar al método de la clase padre usando super()
        study = super().optimize_with_optuna(objective, n_trials=n_trials, direction="maximize")
        return study

    def run_basic_experiment(self):
        """
        Execute the HDBSCAN clustering process on the dataset.

        This method performs HDBSCAN clustering across a range of cluster values (k) 
        from 2 to 9. For each value of k, it calculates the silhouette and Davies-Bouldin 
        scores, saves plots for each clustering configuration, and stores the results.
        
        Side Effects
        ------------
        - Saves clustering plots for each configuration in `folder_plots`.
        - Saves score plots and CSV files in `folder_results`.
        """
        

        # List of metrics to iterate over
        metrics = ["euclidean", "manhattan", "chebyshev"]
        for metric in metrics:
            # Get results for every metric
            k_ = []
            silhouette_coefficients = []
            davies_bouldin_coefficients = []
            
            for min_cluster_size in range(50, 150, 5):  # Adjust range and step as needed
                # Define model parameters
                params = {
                    "metric": metric,
                    "min_cluster_size": min_cluster_size,
                    "min_samples": int(min_cluster_size / 2)
                }
                
                # Run HDBSCAN
                hdbscan_model = hdbscan.HDBSCAN(**params).fit(self.data)
                labels = hdbscan_model.labels_

                # Calculate the number of clusters (excluding noise)
                # TODO: Try Not excluding noise and compare
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                k_.append(n_clusters)
                
                # CALCULATE RESULT DIRS (PLOTTING)
                file_path_plot = os.path.join(self.folder_plots, f"n_clusters_{n_clusters}/metric_{metric}/min_cluster_size_{min_cluster_size}/plot.png")

                # REPRESENTATION
                if n_clusters > 0:
                    # Reduce data to 2D for visualization and save plot
                    # HDBSCAN implements soft clustering, which are examples of
                    # each cluster, and can be a list of examples. So we need to 
                    # calculate at least the most representative por each cluster
                    unique_labels = np.unique(labels)
                    centers = []
                    for label in unique_labels:
                        if label != -1:  # Ignorar puntos de ruido
                            cluster_points = self.data.values[labels == label]
                            cluster_center = np.mean(cluster_points, axis=0)
                            centers.append(cluster_center)

                    if centers:
                        centers = np.array(centers)
                        
                        # Find the 3 nearest neighbors for each centroid and save it in file
                        super().find_and_save_clustering_knn_points(
                            n_neighbors=3, 
                            metric=params.get("metric", "euclidean"),
                            cluster_centers=centers,
                            save_path = os.path.join(self.folder_plots, f"n_clusters_{n_clusters}/metric_{metric}/min_cluster_size_{min_cluster_size}/knn_points/points.csv")
                        )

                        pca_df, pca_centers = super().do_PCA_for_representation(self.data, centers)
                        super().save_clustering_plot(pca_df, labels, pca_centers, i=0, j=1, save_path=file_path_plot)
                    
                    # SCORERS
                    score_silhouette = silhouette_score(self.data, labels) if n_clusters > 1 else 0
                    score_davies = davies_bouldin_score(self.data, labels) if n_clusters > 1 else 99
                    silhouette_coefficients.append(score_silhouette)
                    davies_bouldin_coefficients.append(score_davies)
                else:
                    # Handle cases where no clusters are found
                    silhouette_coefficients.append(0)
                    davies_bouldin_coefficients.append(99)
            
            if n_clusters > 0:

                # CALCULATE RESULT DIRS (RESULTS)
                file_path_result_csv = os.path.join(self.folder_results, f"metric_{metric}/result.csv")
                file_path_result_plot = os.path.join(self.folder_results, f"metric_{metric}/result.png")

                # Save the scores and generate plots
                super().save_clustering_result(
                    k_, 
                    silhouette_coefficients, 
                    davies_bouldin_coefficients, 
                    [],
                    # paths
                    file_path_result_plot,
                    file_path_result_csv,
                    ""
                )


if __name__ == "__main__":
    # Test the KMeansClustering class with a sample dataset
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    data = pd.DataFrame(data, columns=['x', 'y'])
    kmeans_clustering = HDBSCANClustering(data)
    kmeans_clustering.run()
    print("HDBSCAN clustering complete. Results and plots saved.")
