from collections import Counter
from itertools import product
import os
from pathlib import Path
import pickle
import sys
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_factory import ClusteringFactory
from src.clustering.clustering_model import ClusteringModel
from src.preprocess.preprocess import Preprocess


class ExperimentResultController():

    def __init__(self, 
                 eval_method="silhouette",
                 cache= True, 
                 verbose= False,
                 **kwargs):
    
        # Setup attrs
        self.eval_method = eval_method
        self.cache = cache
        self.verbose = verbose

        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        # Dir where all experiments are stored
        self.results_dir = (
            Path(__file__).resolve().parent
            / f"results"
        )
        # Plot dir
        self.plot_dir = (
            Path(__file__).resolve().parent
            / f"plots"
        )
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Load all experiments for given eval_method
        self.results_df = None
        self.__load_all_experiments()


    
    def __load_all_experiments(self):
        """
        Loads all experiment of given eval_method. It does not care if it is hdbscan, optuna, gridsearch,
        etc. We are gonna be loading the bests
        """
        logger.info(f"LOADING ALL EXPERIMENTS OF EVAL METHOD {self.eval_method.upper()}")
        
        if self.eval_method not in ("silhouette", "davies_bouldin"):
            raise ValueError("Eval method not supported")
        
        dataframes = []
        for file_path in self.results_dir.rglob('*.pkl'):
            if "silhouette" in file_path.parts:
                with open(file_path, "rb") as file:
                    try:
                        df = pickle.load(file)
                        # Ensure the DataFrame has the "eval_method" column and matches the specified method
                        if df.get("eval_method") == self.eval_method:
                            dataframes.append(df)
                    except Exception as e:
                        logger.warning(f"Could not load {file_path}: {e}")

        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
            self.results_df = merged_df
            logger.info("Experiments successfully loaded and merged.")
        else:
            logger.info("No experiments found with the specified eval_method.")
            self.results_df = None



    def add_path_type(self, file_suffix):
        """
        Helper to create file paths for saving plots.
        """
        return str(self.plot_dir / f"{file_suffix}.png")



    def get_top_k_experiments(self, top_k: int, min_n_cluster: int, min_dimension: int, best_silhouette_noise_ratio: bool) -> pd.DataFrame:
        """
        Returns the top_k experiments based on the specified criteria.

        Parameters:
            top_k (int): Number of top experiments to return.
            min_n_cluster (int): Minimum number of clusters.
            min_dimension (int): Minimum dimension for reduced data.
            best_silhouette_noise_ratio (bool): If True, sort by silhouette_noise_ratio; otherwise, sort by best_value_w/o_penalty.

        Returns:
            pd.DataFrame: Filtered DataFrame with the top_k experiments.
        """
        # Verify df is loaded
        if self.results_df is None:
            logger.warning("No experiments loaded. Returning an empty DataFrame.")
            return pd.DataFrame()

        # Filter dataframe
        filtered_df = self.results_df[
            (self.results_df['n_clusters'] >= min_n_cluster) & 
            (self.results_df['dimensions'] >= min_dimension)
        ]

        # Ordena el DataFrame por la columna deseada
        if best_silhouette_noise_ratio:
            sorted_df = filtered_df.sort_values(by='silhouette_noise_ratio', ascending=False)
        else:
            sorted_df = filtered_df.sort_values(by='best_value_w/o_penalty', ascending=False)

        # Selecciona los top_k experimentos
        top_k_df = sorted_df.head(top_k)

        return top_k_df


    def get_experiment_data(self, filtered_df, experiment_type):
        """
        Helper function to get the best experiment based on `experiment_type`.

        Parameters
        ----------
        experiment_type : str
            "best" for best silhouette or "silhouette_noise_ratio" for best silhouette-to-noise ratio.

        Returns
        -------
        pd.Series
            The row in the DataFrame corresponding to the best experiment.
        """
        # First of all, filter those with cluster number less than 10 for example
        # This should be an input parameter 
        
        if filtered_df.empty:
            raise ValueError("No experiments found.")

        if experiment_type == "best":
            return filtered_df.loc[filtered_df[self.string_silhouette].idxmax()]
        elif experiment_type == "silhouette_noise_ratio":
            return filtered_df.loc[filtered_df["silhouette_noise_ratio"].idxmax()]
        else:
            raise ValueError("Invalid experiment type. Choose 'best' or 'silhouette_noise_ratio'.")





    def show_best_silhouette(self, experiment="best", show_all=False, top_n=15, min_clusters=30, show_cluster_index=False, show_plots=False):
        """
        Displays the top `top_n` clusters with the highest silhouette average and the 
        `top_n` clusters with the lowest silhouette average, only if the total cluster 
        count exceeds `min_clusters`. If there are `min_clusters` or fewer clusters, 
        it displays all clusters without filtering.

        Parameters
        ----------
        top_n : int
            Number of clusters to display in the sections for highest and lowest silhouette average.
        min_clusters : int
            Minimum number of clusters required to apply filtering for the best and worst clusters.
        """
        # Check if results_df contains results
        if self.experiment.results_df is None or self.experiment.results_df.empty:
            logger.warning("No results found in the experiment DataFrame.")
            return

        # Get the experiment data based on the specified `experiment` type
        best_experiment = self.get_experiment_data(experiment)
 

        # Extract information for the best configuration
        best_labels = best_experiment['labels']
        scaler = best_experiment['scaler']
        dim_reduction = best_experiment['dim_reduction']
        dimensions = best_experiment['dimensions']
        params = best_experiment['best_params']
        optimizer = best_experiment['optimization']
        original_silhouette_score = best_experiment[self.string_silhouette]
        
        # Get scaled and reduced data for the best configuration
        scaled_data = self.experiment._eda.run_scaler(scaler)
        reduced_data = self.experiment._eda.run_dim_red(
            scaled_data, dimensions=dimensions, dim_reduction=dim_reduction, show_plots=False
        )

        # Exclude noise points with label -1
        non_noise_mask = best_labels != -1
        non_noise_labels = best_labels[non_noise_mask]
        non_noise_data = reduced_data[non_noise_mask]

        # Calculate silhouette values for each non-noise point
        silhouette_values = silhouette_samples(non_noise_data, non_noise_labels)

        # Calculate average silhouette per cluster
        unique_labels = np.unique(non_noise_labels)
        cluster_count = len(unique_labels)

        # If there are `min_clusters` or fewer clusters, plot all clusters
        if cluster_count <= min_clusters or show_all:
            logger.info(f"The number of clusters ({cluster_count}) is less than or equal to {min_clusters}. "
                        "All clusters will be plotted.")
            selected_clusters = unique_labels
        else:
            # Compute the average silhouette for each cluster
            cluster_silhouette_means = {
                label: silhouette_values[non_noise_labels == label].mean() for label in unique_labels
            }

            # Select the top `top_n` clusters with the best and worst silhouette averages
            top_clusters = sorted(cluster_silhouette_means, key=cluster_silhouette_means.get, reverse=True)[:top_n]
            bottom_clusters = sorted(cluster_silhouette_means, key=cluster_silhouette_means.get)[:top_n]

            # Combine the top and bottom clusters and ensure uniqueness, ordering from lowest to highest silhouette
            selected_clusters = sorted(set(top_clusters + bottom_clusters), key=lambda label: cluster_silhouette_means[label])

        # Generate the silhouette plot, ordered from lowest to highest average silhouette score
        plt.figure(figsize=(10, 7))
        y_lower = 10
        for i, label in enumerate(selected_clusters):
            ith_cluster_silhouette_values = silhouette_values[non_noise_labels == label]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
            # Show cluster index if specified
            if show_cluster_index:
                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))

            y_lower = y_upper + 10

        # Add a vertical line for the original silhouette score from the experiment
        plt.axvline(x=original_silhouette_score, color="red", linestyle="--", label=f"Original Silhouette Score: {original_silhouette_score:.2f}")
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster Index")
        plt.title(f"Silhouette Coefficient Plot for Best Configuration {optimizer}\n\n"
                f"Clustering: {self.experiment.clustering} - Dimensionality Reduction: {dim_reduction} - Dimensions: {dimensions}\n"
                f"Params: {params}\n"
                )
        plt.legend()

        # Save the plot
        file_suffix = "best_trial_silhouette" if experiment == "best" else "sil_noise_ratio_trial_silhouette"
        plt.savefig(self.add_path_type(file_suffix), bbox_inches='tight')
        if show_plots:
            plt.show()

        logger.info(f"Scatter plot generated for the selected experiment ({experiment}).")





if __name__ == "__main__":
    pass
