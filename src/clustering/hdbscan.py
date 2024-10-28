from datetime import datetime
import hdbscan
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
        
        

    def run(self):
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
        
        # Collect indexes
         # Initialize lists to store results
        k_ = []
        silhouette_coefficients = []
        davies_boulding_coefficients = []

        # List of metrics to iterate over
        metrics = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]
        # TODO: PARA CADA MÉTRICA HAY QUE GENERAR UN ARCHIVO CON LAS Ks Y results
        # Iterate over each metric and different values of min_cluster_size
        for metric in metrics:
            for min_cluster_size in range(50, 150, 5):  # Adjust range and step as needed
                # Define model parameters
                params = {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": int(min_cluster_size / 2),
                    "metric": metric,
                }
                # Generate a unique identifier for the current parameters
                param_string = "__".join([f"{key}_{value}" for key, value in params.items()])
                file_path_plot = os.path.join(self.folder_plots, param_string) + ".png"

                if metric == "mahalanobis":
                    cov_matrix = np.cov(self.data.values, rowvar=False)
                    VI = np.linalg.inv(cov_matrix)
                    params["VI"] = VI
                

                # Run HDBSCAN
                hdbscan_model = hdbscan.HDBSCAN(**params).fit(self.data)
                labels = hdbscan_model.labels_

                # Calculate the number of clusters (excluding noise)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                k_.append(n_clusters)

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
                        pca_df, pca_centers = super().do_PCA_for_representation(self.data, centers)
                        super().save_clustering_plot(pca_df, labels, pca_centers, i=0, j=1, save_path=file_path_plot)
                    
                    # SCORERS
                    score_silhouette = silhouette_score(self.data, labels) if n_clusters > 1 else 0
                    score_davies = davies_bouldin_score(self.data, labels) if n_clusters > 1 else 99
                    silhouette_coefficients.append(score_silhouette)
                    davies_boulding_coefficients.append(score_davies)
                else:
                    # Handle cases where no clusters are found
                    silhouette_coefficients.append(0)
                    davies_boulding_coefficients.append(99)

        # Save the scores and generate plots
        super().save_clustering_result(
            k_, silhouette_coefficients, davies_boulding_coefficients, []
        )


if __name__ == "__main__":
    # Test the KMeansClustering class with a sample dataset
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    data = pd.DataFrame(data, columns=['x', 'y'])
    kmeans_clustering = HDBSCANClustering(data)
    kmeans_clustering.run()
    print("HDBSCAN clustering complete. Results and plots saved.")
