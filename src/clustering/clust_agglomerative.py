from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_model import ClusteringModel


class AgglomerativeClusteringModel(ClusteringModel):
    """
    Agglomerative Clustering model class inheriting from ClusteringModel.

    This class implements the Agglomerative Clustering algorithm on a dataset 
    and provides methods to run clustering, calculate metrics, and save results 
    including plots and score data.
    """

    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Initialize the AgglomerativeClusteringModel.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be clustered.
        **kwargs : dict
            Additional parameters for customization or future expansion.
        """
        super().__init__(data, model_name="agglomerative")
  
    
    
    def run_optuna(self, evaluation_method="silhouette", n_trials=50):
        """
        Run Optuna optimization for the Agglomerative Clustering model with a specified evaluation method.

        This method sets up and executes an Optuna hyperparameter optimization for the Agglomerative 
        Clustering algorithm. It defines the range of hyperparameters specific to Agglomerative Clustering, 
        including `n_clusters`, `linkage`, and `metric`. The optimization process seeks to maximize the 
        silhouette score or minimize the Davies-Bouldin score, depending on the selected `evaluation_method`.
        
        Parameters
        ----------
        evaluation_method : str, optional
            The evaluation metric to optimize. It can be either 'silhouette' (to maximize the silhouette score) 
            or 'davies_bouldin' (to minimize the Davies-Bouldin score). Defaults to "silhouette".
        n_trials : int, optional
            The number of optimization trials to run. Defaults to 50.

        Returns
        -------
        optuna.study.Study
            The Optuna study object containing details of the optimization process, including the best 
            hyperparameters found and associated evaluation score.
        
        Notes
        -----
        - This method calls the generic `run_optuna_generic` method from the base class `ClusteringModel`, 
        which manages the Optuna optimization process and the model evaluation.
        - `model_builder` is a nested function that constructs an Agglomerative Clustering model using 
        hyperparameters suggested by each Optuna trial. Note that if `linkage` is set to 'ward', 
        the `metric` is automatically set to 'euclidean' as required by the Agglomerative Clustering algorithm.
        """
         # Param/model builder for Agglomerative
        def model_builder(trial):
            n_clusters = trial.suggest_int('n_clusters', 2, 10)
            linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
            metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine'])
            custom_metric = metric if linkage != "ward" else "euclidean"
            
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=custom_metric
            )

         # Call generic class method
        return self.run_optuna_generic(model_builder, evaluation_method, n_trials)
        

    def run_basic_experiment(self):
        """
        Execute the Agglomerative Clustering process on the dataset.

        This method performs Agglomerative Clustering across a range of cluster 
        values from 2 to 9, and iterates over different linkage and metric combinations.
        
        Side Effects
        ------------
        - Saves clustering plots for each configuration in `folder_plots`.
        - Saves score plots and CSV files in `folder_results`.
        """
        
        # Linkage and metric types
        linkages = ["ward", "complete", "average", "single"]
        metrics = ["euclidean", "manhattan", "cosine"]

        for linkage in linkages:
            for metric in metrics:
                
                # Ensure 'ward' linkage only uses 'euclidean' metric
                custom_metric = metric if linkage != "ward" else "euclidean"
                
                k_ = []
                silhouette_coefficients = []
                davies_bouldin_coefficients = []

                for n_clusters in range(2, 9):  # Number of clusters to try
                    params = {
                        "linkage": linkage,
                        "metric": custom_metric,
                        "n_clusters": n_clusters
                    }
                    
                    # Run AgglomerativeClustering
                    clustering_model = AgglomerativeClustering(**params).fit(self.data)
                    labels = clustering_model.labels_

                    # Count clusters (ignoring noise, if any)
                    unique_labels = np.unique(labels)
                    n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
                    k_.append(n_clusters_found)

                    # Define file paths
                    file_path_plot = os.path.join(self.folder_plots, f"n_clusters_{n_clusters}/linkage_{linkage}/metric_{metric}/plot.png")

                    # If clusters found, save representative plots
                    if n_clusters_found > 1:
                        unique_labels = np.unique(labels)
                        centers = []
                        for label in unique_labels:
                            cluster_points = self.data.values[labels == label]
                            cluster_center = np.mean(cluster_points, axis=0)
                            centers.append(cluster_center)

                        if centers:
                            centers = np.array(centers)

                            # Save nearest neighbors of centers
                            super().find_and_save_clustering_knn_points(
                                n_neighbors=3, 
                                metric=custom_metric,
                                cluster_centers=centers,
                                save_path=os.path.join(self.folder_plots, f"n_clusters_{n_clusters}/linkage_{linkage}/metric_{metric}/knn_points/points.csv")
                            )

                            pca_df, pca_centers = super().do_PCA_for_representation(self.data, centers)
                            super().save_clustering_plot(pca_df, labels, pca_centers, i=0, j=1, save_path=file_path_plot)
                        
                        # Calculate and save scores
                        score_silhouette = silhouette_score(self.data, labels) if n_clusters > 1 else 0
                        score_davies = davies_bouldin_score(self.data, labels) if n_clusters > 1 else 99
                        silhouette_coefficients.append(score_silhouette)
                        davies_bouldin_coefficients.append(score_davies)
                    else:
                        silhouette_coefficients.append(0)
                        davies_bouldin_coefficients.append(99)

                # Save clustering results
                if k_:
                    file_path_result_csv = os.path.join(self.folder_results, f"linkage_{linkage}/metric_{metric}/result.csv")
                    file_path_result_plot = os.path.join(self.folder_results, f"linkage_{linkage}/metric_{metric}/result.png")

                    # Save scores and generate plots
                    super().save_clustering_result(
                        k_, 
                        silhouette_coefficients, 
                        davies_bouldin_coefficients, 
                        [],
                        file_path_result_plot,
                        file_path_result_csv,
                        ""
                    )


if __name__ == "__main__":
    # Test the AgglomerativeClusteringModel class with a sample dataset
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    data = pd.DataFrame(data, columns=['x', 'y'])
    agglomerative_clustering = AgglomerativeClusteringModel(data)
    agglomerative_clustering.run()
    print("Agglomerative clustering complete. Results and plots saved.")
