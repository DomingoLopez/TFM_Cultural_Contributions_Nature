from collections import Counter
from itertools import product
import os
from pathlib import Path
import pickle
import sys
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
        experiment_files = Path(self.results_dir).rglob("*.pkl")
        experiments = []
        for file in experiment_files:
            try:
                with open(file, "rb") as f:
                    result = pickle.load(f)
                    
                # Check if the loaded result is a valid DataFrame
                if isinstance(result, pd.DataFrame) and not result.empty:
                    experiments.append(result)
                else:
                    logger.warning(f"Invalid or empty result file: {file}")
            
            except Exception as e:
                logger.warning(f"Could not load {file}: {e}")

        # Combine all valid results
        if experiments:
            self.results_df = pd.concat(experiments, ignore_index=True)
        else:
            self.results_df = pd.DataFrame()
            logger.warning("No experiments found with the specified eval_method.")



    def add_path_type(self, file_suffix):
        """
        Helper to create file paths for saving plots.
        """
        return str(self.plot_dir / f"{file_suffix}.png")



    def get_top_k_experiments(self, top_k: int, 
                              n_cluster_range: tuple,
                              reduction_params: dict,
                              use_score_noise_ratio: bool) -> pd.DataFrame:
        """
        Returns the top_k experiments based on the specified criteria.

        Parameters:
            top_k (int): Number of top experiments to return.
            min_n_cluster (int): Minimum number of clusters.
            max_n_cluster (int): Maximum number of clusters.
            min_dimension (int): Minimum dimension for reduced data.
            max_dimension (int): Maximum dimension for reduced data.
            use_score_noise_ratio (bool): If True, sort by score_noise_ratio; otherwise, sort by score_w/o_penalty.

        Returns:
            pd.DataFrame: Filtered DataFrame with the top_k experiments.
        """
        
        # Validate n_cluster_range
        min_n_cluster, max_n_cluster = n_cluster_range
        if min_n_cluster < 2 or max_n_cluster > 800:
            raise ValueError("n_cluster_range values must be between 2 and 800.")
        if min_n_cluster > max_n_cluster:
            raise ValueError("min_n_cluster cannot be greater than max_n_cluster.")
        
        # Validate reduction_params
        for key, value_range in reduction_params.items():
            if not isinstance(value_range, tuple) or len(value_range) != 2:
                raise ValueError(f"Parameter {key} in reduction_params must be a tuple (min, max).")
            if value_range[0] > value_range[1]:
                raise ValueError(f"Invalid range for {key}: {value_range}. Min cannot be greater than Max.")
    
            


        
        # Verify df is loaded
        if self.results_df is None:
            logger.warning("No experiments loaded. Returning an empty DataFrame.")
            return pd.DataFrame()

        # Filter dataframe based on cluster
        filtered_df = self.results_df[
            (self.results_df['n_clusters'] >= min_n_cluster) & 
            (self.results_df['n_clusters'] <= max_n_cluster) 
        ]

        # Filter by reduction params
        for param, value_range in reduction_params.items():
            min_val, max_val = value_range
            filtered_df = filtered_df[
                filtered_df['reduction_params'].apply(
                    lambda params: param in params and min_val <= params[param] <= max_val
                )
            ]

        # Determine sorting column and order based on eval_method
        if self.eval_method == "davies_bouldin":
            sort_column = 'score_noise_ratio' if use_score_noise_ratio else 'score_w/o_penalty'
            ascending_order = True  # Lower is better for davies_bouldin
        else:
            sort_column = 'score_noise_ratio' if use_score_noise_ratio else 'score_w/o_penalty'
            ascending_order = False  # Higher is better for silhouette

        # Sort the DataFrame
        sorted_df = filtered_df.sort_values(by=sort_column, ascending=ascending_order)

        # Select the top_k experiments
        top_k_df = sorted_df.head(top_k)
        
        return top_k_df


    def get_best_experiment_data(self, filtered_df, use_score_noise_ratio):
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

        if (use_score_noise_ratio):
            df = filtered_df.loc[filtered_df["score_noise_ratio"].idxmax()] if self.eval_method == "silhouette" else filtered_df.loc[filtered_df["score_noise_ratio"].idxmin()]
            logger.info(f"Selected experiment with score/noise ratio: {df['score_noise_ratio']}")
        else:
            df = filtered_df.loc[filtered_df["score_w/o_penalty"].idxmax()] if self.eval_method == "silhouette" else filtered_df.loc[filtered_df["score_w/o_penalty"].idxmin()]
            logger.info(f"Selected experiment with score: {df['score_w/o_penalty']}")
            
        return df


    def show_best_silhouette(self, experiments = None, use_score_noise_ratio=True, show_all=False, top_n=25, min_clusters=50, show_cluster_index=False, show_plots=False):
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
        
        experiments = experiments if experiments is not None else self.results_df
        
        # Check if results_df contains results
        if experiments is None or experiments.empty:
            logger.warning("No results found in the experiment DataFrame.")
            return

        # Get the experiment data based on the specified `experiment` type
        best_experiment = self.get_best_experiment_data(experiments, use_score_noise_ratio)
 
        # Extract information for the best configuration
        best_id = best_experiment['id']
        best_index = best_experiment.index[0] if hasattr(best_experiment.index, '__iter__') else best_experiment.index
        best_labels = best_experiment['labels']
        clustering = best_experiment['clustering']
        scaler = best_experiment['scaler']
        dim_red = best_experiment['dim_red']
        dimensions = best_experiment['dimensions']
        params = best_experiment['best_params']
        optimizer = best_experiment['optimization']
        original_score = best_experiment['score_w/o_penalty']
        embeddings_used = best_experiment['embeddings']



        # Exclude noise points (label -1)
        non_noise_mask = best_labels != -1
        non_noise_labels = best_labels[non_noise_mask]
        non_noise_data = embeddings_used[non_noise_mask]

        # Calculate silhouette values for non-noise data
        silhouette_values = silhouette_samples(non_noise_data, non_noise_labels)

        # Calculate average silhouette per cluster
        unique_labels = np.unique(non_noise_labels)
        cluster_count = len(unique_labels)


        # Select clusters to display based on min_clusters and top_n
        if cluster_count <= min_clusters or show_all:
            logger.info(f"The number of clusters ({cluster_count}) is less than or equal to {min_clusters} OR SHOW_ALL is set to True. "
                        "All clusters will be plotted.")
            selected_clusters = unique_labels
        else:
            # Calculate silhouette averages for each cluster
            cluster_silhouette_means = {
                label: silhouette_values[non_noise_labels == label].mean() for label in unique_labels
            }

            # Select the top `top_n` clusters with best and worst silhouette averages
            top_clusters = sorted(cluster_silhouette_means, key=cluster_silhouette_means.get, reverse=True)[:top_n]
            bottom_clusters = sorted(cluster_silhouette_means, key=cluster_silhouette_means.get)[:top_n]

            # Combine the top and bottom clusters
            selected_clusters = sorted(set(top_clusters + bottom_clusters), key=lambda label: cluster_silhouette_means[label])

        # Generate silhouette plot
        plt.figure(figsize=(10, 7))
        y_lower = 10
        for i, label in enumerate(selected_clusters):
            ith_cluster_silhouette_values = silhouette_values[non_noise_labels == label]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7, label=f"Cluster {label}")
            if show_cluster_index:
                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
            y_lower = y_upper + 10

        # Add a vertical line for the original silhouette score
        plt.axvline(x=original_score, color="red", linestyle="--", label=f"Original Score: {original_score:.2f}")
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster Index")
        plt.title(f"Silhouette Plot for Best Configuration {optimizer}\n"
                f"Clustering: {clustering} | Dim Reduction: {dim_red} | Dimensions: {dimensions}\n"
                f"Params: {params}")
        plt.legend()

        # Save the plot
        file_suffix = "best_silhouette" if use_score_noise_ratio else "silhouette_noise_ratio"
        file_path = os.path.join(self.plot_dir, str(best_id),f"index_{best_index}_silhouette_{original_score:.3f}_{file_suffix}.png")
        os.makedirs(os.path.join(self.plot_dir, str(best_id)), exist_ok=True)
        plt.savefig(file_path, bbox_inches='tight')
        if show_plots:
            plt.show()

        logger.info(f"Silhouette plot saved to {file_path}.")






    def show_best_scatter(self,  experiments = None, use_score_noise_ratio=True, show_all=False, show_plots=False):
        """
        Plots a 2D scatter plot for the best experiment configuration, with clusters reduced 
        to 2D space using PCA and color-coded for better visual distinction. Points labeled 
        as noise (-1) are always shown in red.
        """
        
        experiments = experiments if experiments is not None else self.results_df
        # Check if results_df contains results
        if experiments is None or experiments.empty:
            logger.warning("No results found in the experiment DataFrame.")
            return
        # Get the experiment data based on the specified `experiment` type
        best_experiment = self.get_best_experiment_data(experiments, use_score_noise_ratio)
        best_index = best_experiment.index[0] if hasattr(best_experiment.index, '__iter__') else best_experiment.index
        best_id = best_experiment['id']
        best_labels = np.array(best_experiment['labels'])
        optimizer = best_experiment['optimization']
        clustering = best_experiment['clustering']
        scaler = best_experiment['scaler']
        dim_red = best_experiment['dim_red']
        dimensions = best_experiment['dimensions']
        params = best_experiment['best_params']
        embeddings = best_experiment['embeddings']
        original_score = best_experiment['score_w/o_penalty']
        cluster_count = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)  # Exclude noise (-1) from cluster count


        # TODO THIS
        # Get data reduced from eda object
        data = embeddings.values

        # Check if reduction is needed
        if data.shape[1] > 2:
            # Reduce dimensions with PCA
            pca = PCA(n_components=2, random_state=42)
            reduced_data = pca.fit_transform(data)
        else:
            # Use the data directly if already 2D
            reduced_data = data

        # Define colormap for clusters and manually assign red for noise
        colors = sns.color_palette("viridis", cluster_count)
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(12, 9))
        
        # Plot noise points (label -1) in red
        noise_points = reduced_data[best_labels == -1]
        plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', s=10, alpha=0.6, label="Noise (-1)")
        
        # Plot cluster points
        cluster_points = reduced_data[best_labels != -1]
        cluster_labels = best_labels[best_labels != -1]
        scatter = plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_labels, cmap=cmap, s=10, alpha=0.6)

        # Add colorbar if useful to distinguish clusters
        plt.colorbar(scatter, spacing="proportional", ticks=np.linspace(0, cluster_count, num=10))
        
        plt.title(f"Scatter Plot of Best Experiment - {optimizer} (Noise in Red, Clusters in 2D PCA) \n\n"
                f"Clustering: {clustering} - Dim Reduction: {dim_red} - Dimensions: {dimensions}\n"
                f"Params: {params}\n"
                )
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # Save and show plot
        file_suffix = "best_scatter" if use_score_noise_ratio else "best_scatter_noise_ratio"
        file_path = os.path.join(self.plot_dir, str(best_id),f"index_{best_index}_silhouette_{original_score:.3f}_{file_suffix}.png")
        os.makedirs(os.path.join(self.plot_dir, str(best_id)), exist_ok=True)
        plt.savefig(file_path, bbox_inches='tight')

        if show_plots:
            plt.show()
        logger.info(f"Scatter plot generated for the selected experiment saved to {file_path}.")




if __name__ == "__main__":
    pass
