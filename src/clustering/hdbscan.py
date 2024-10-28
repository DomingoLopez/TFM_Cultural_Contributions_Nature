from datetime import datetime
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
        super().__init__(data, model_name="kmeans")
        
        

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
        error = []
        k_ = []
        silhouette_coefficients = []
        davies_boulding_coefficients = []

        # Perform clustering for each k
        for k in np.arange(2, 10):
            # Define params
            params = {
                "n_clusters": k,
                "n_init": 100,
                "init": "k-means++",
                "random_state": 1234
            }
            # Define file paths for saving plots and results
            param_string = "__".join([f"{key}_{value}" for key, value in params.items()])
            file_path_plot = os.path.join(self.folder_plots, param_string) + ".png"
            file_path_result_plot = os.path.join(self.folder_results, param_string) + "_result.png"
            file_path_result_csv = os.path.join(self.folder_results, param_string) + "_result.csv"
            file_path_error = os.path.join(self.folder_results, param_string) + "_error.png"

            # Run KMeans
            kmeans = KMeans(n_clusters=params["n_clusters"], init=params["init"], 
                            n_init=params["n_init"], random_state=params["random_state"]).fit(self.data)
            error.append(kmeans.inertia_)
            
            # REPRESENTATION
            pca_df, pca_centers = super().do_PCA_for_representation(self.data, kmeans.cluster_centers_)
            super().save_clustering_plot(pca_df, kmeans.labels_, pca_centers, i=0, j=1, save_path=file_path_plot)

            # SCORERS
            score_silhouette = silhouette_score(self.data, kmeans.labels_)
            score_davies = davies_bouldin_score(self.data, kmeans.labels_)
            silhouette_coefficients.append(score_silhouette)
            davies_boulding_coefficients.append(score_davies)
            k_.append(k)

        # Save the scores and generate plots
        super().save_clustering_result(
            k_, silhouette_coefficients, davies_boulding_coefficients, error,
            save_path=file_path_result_plot, 
            save_path_csv=file_path_result_csv, 
            save_path_error = file_path_error
        )


if __name__ == "__main__":
    # Test the KMeansClustering class with a sample dataset
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    data = pd.DataFrame(data, columns=['x', 'y'])
    kmeans_clustering = HDBSCANClustering(data)
    kmeans_clustering.run()
    print("HDBSCAN clustering complete. Results and plots saved.")
