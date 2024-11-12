import os
from pathlib import Path
import pickle
import sys
import seaborn as sns
from typing import Optional, Tuple
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_factory import ClusteringFactory
from src.clustering.clustering_model import ClusteringModel
from src.preprocess.preprocess import EDA
from src.experiment.experiment import Experiment


class ClusteringPlot():
    """
    Class for plotting clusters and results from Clustering experiments.
    
    """

    def __init__(self, 
            experiment: Experiment
        ):

        # Setup attrs
        self.experiment = experiment

        # Common attrs
        self.string_silhouette = "best_value_w/o_penalty" if self.experiment.optimizer == "optuna" else "value_w/o_penalty"

        
        # Dirs and files
        self.main_plot_dir = Path(__file__).resolve().parent / \
                                f"plots/{self.experiment.clustering}" / \
                                ("optuna" if self.experiment.optimizer == "optuna" else "gridsearch") / \
                                f"dim_red_{self.experiment.dim_reduction}/{self.experiment.eval_method}_penalty_{self.experiment.penalty}" 
        os.makedirs(self.main_plot_dir, exist_ok=True)

    
    
    def add_path_type(self,type):
        """_summary_

        Parameters
        ----------
        type : str
            Type of plot to save

        Returns
        -------
        path : str
            Saving path
            
        """
        return os.path.join(self.main_plot_dir,f"{type}.png")
        



    def get_experiment_data(self, experiment_type):
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
        if experiment_type == "best":
            return self.experiment.results_df.loc[self.experiment.results_df[self.string_silhouette].idxmax()]
        elif experiment_type == "silhouette_noise_ratio":
            return self.experiment.results_df.loc[self.experiment.results_df["silhouette_noise_ratio"].idxmax()]
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

    
    
    
    
    def show_best_scatter(self, experiment="best", show_plots=False):
        """
        Plots a 2D scatter plot for the best experiment configuration, with clusters reduced 
        to 2D space using PCA and color-coded for better visual distinction. Points labeled 
        as noise (-1) are always shown in red.
        """
    
        
        # Get the experiment data based on the specified `experiment` type
        best_experiment = self.get_experiment_data(experiment)
        best_labels = np.array(best_experiment['labels'])
        optimizer = best_experiment['optimization']
        scaler = best_experiment['scaler']
        dim_reduction = best_experiment['dim_reduction']
        dimensions = best_experiment['dimensions']
        params = best_experiment['best_params']
        cluster_count = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)  # Exclude noise (-1) from cluster count

        # Get data reduced from eda object
        data = self.experiment.eda.check_reduced_exists_cache(scaler, dim_reduction, dimensions).values

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
                f"Clustering: {self.experiment.clustering} - Dim Reduction: {dim_reduction} - Dimensions: {dimensions}\n"
                f"Params: {params}\n"
                )
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # Save and show plot
        file_suffix = "best_experiment_scatter" if experiment == "best" else "sil_noise_ratio_experiment_scatter"
        plt.savefig(self.add_path_type(file_suffix), bbox_inches='tight')
        if show_plots:
            plt.show()
        logger.info(f"Scatter plot generated for the selected experiment ({experiment}).")

        
    
    
    
    
    
    def show_best_scatter_with_centers(self, experiment="best", show_plots=False):
        """
        Plots a 2D scatter plot for the best experiment configuration, with clusters reduced 
        to 2D space using PCA and color-coded for better visual distinction. Points labeled 
        as noise (-1) are always shown in red.
        """
        
        # Get the experiment data based on the specified `experiment` type
        best_experiment = self.get_experiment_data(experiment)
        # TODO Need 
        best_labels = np.array(best_experiment['labels'])
        # Convert embeddings and centers to numpy arrays if they are DataFrames
        best_centers = best_experiment['centers'].values if isinstance(best_experiment['centers'], pd.DataFrame) else np.array(best_experiment['centers'])
        optimizer = best_experiment['optimization']
        dim_reduction = best_experiment['dim_reduction']
        dimensions = best_experiment['dimensions']
        scaler = best_experiment['scaler']
        params = best_experiment['best_params']
        cluster_count = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)  # Exclude noise (-1) from cluster count
        # Get data reduced from eda object
        data = self.experiment.eda.check_reduced_exists_cache(scaler, dim_reduction, dimensions).values

        # Check if reduction is needed
        if data.shape[1] > 2:
            # Reduce dimensions with PCA
            pca = PCA(n_components=2, random_state=42)
            reduced_data = pca.fit_transform(data)
            pca_centers = pca.transform(best_centers)
        else:
            # Use the data directly if already 2D
            reduced_data = data
            pca_centers = best_centers

        # Color mapping for clusters and plot setup
        colors = ['#00FF00', '#FFFF00', '#0000FF', '#FF9D0A', '#00B6FF', '#F200FF', '#FF6100']
        cmap_bold = ListedColormap(colors)
        plt.figure(figsize=(9, 6))
        
        # Plot noise points (label -1) in red
        noise_points = reduced_data[best_labels == -1]
        plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', s=10, alpha=0.6, label="Noise (-1)")
        
        # Plot cluster points
        cluster_points = reduced_data[best_labels != -1]
        cluster_labels = best_labels[best_labels != -1]
        scatter = plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_labels, cmap=cmap_bold, s=20, alpha=0.6)

        # Plot cluster centers
        if pca_centers is not None:
            plt.scatter(pca_centers[:, 0], pca_centers[:, 1], marker='D', c='black', s=10, label="Cluster Centers", edgecolors='black')
        
        # Add colorbar to distinguish clusters
        plt.colorbar(scatter, spacing="proportional", ticks=np.arange(0, cluster_count + 1, max(1, cluster_count // 10)))

        
        plt.title(f"Scatter Plot of Best Experiment with Cluster Centers - {optimizer} (Noise in Red) \n\n"
                f"Clustering: {self.experiment.clustering} - Dim Reduction: {dim_reduction} - Dimensions: {dimensions}\n"
                f"Params: {params}\n"
                )
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # Save and show plot
        file_suffix = "best_experiment_with_centers" if experiment == "best" else "sil_noise_ratio_experiment_with_centers"
        plt.savefig(self.add_path_type(file_suffix), bbox_inches='tight')
        if show_plots:
            plt.show()
        logger.info(f"Scatter plot with centers generated for the selected experiment ({experiment}).")



    def show_best_clusters_counters_comparision(self, experiment="best", show_plots=False):
        """
        Displays a bar chart comparing the number of points in each cluster for the best configuration.
        
        The method retrieves the cluster sizes (number of points per cluster) from `label_counter`
        for the best experiment configuration and displays a bar chart to compare cluster sizes.

        Parameters
        ----------
        show_plots : bool, optional
            If True, displays the plot. Default is False.
        """
        # Check if results_df contains results
        if self.experiment.results_df is None or self.experiment.results_df.empty:
            logger.warning("No results found in the experiment DataFrame.")
            return

        # Get the experiment data based on the specified `experiment` type
        best_experiment = self.get_experiment_data(experiment)
        label_counter = best_experiment['label_counter']
        
        if not label_counter:
            logger.warning("No label counter found for the best experiment.")
            return
        
        label_counter_filtered = {k: v for k, v in label_counter.items() if k != -1}

        # Extract cluster indices and their respective counts from label_counter
        cluster_indices = list(label_counter_filtered.keys())
        cluster_sizes = list(label_counter_filtered.values())

        # Count total with noise and without noise
        total_minus_one = label_counter.get(-1, 0)
        total_rest = sum(v for k, v in label_counter.items() if k != -1)
        
        # Plot the bar chart
        plt.figure(figsize=(12, 6))
        sns.barplot(x=cluster_indices, y=cluster_sizes, palette="viridis")
        plt.xlabel("Cluster Index")
        plt.ylabel("Number of Points")
        plt.title("Comparison of Cluster Sizes for Best Experiment\n\n" \
                  f"Total cluster points: {total_rest}\n"   \
                  f"Total noise points: {total_minus_one}\n")
        plt.xticks(rotation=45)
        
        # Save the plot with a name based on the `experiment` type
        file_suffix = "clusters_counter_comparison" if experiment == "best" else "sil_noise_ratio_clusters_counter_comparision"
        plt.savefig(self.add_path_type(file_suffix), bbox_inches='tight')
        if show_plots:
            plt.show()
        logger.info(f"Cluster counter comparison plot generated for the selected experiment ({experiment}).")
    



    def show_top_noise_silhouette(self, top_n=20, priority="eval_method", show_plots=False):
        """
        Sorts results based on the specified priority: either by the highest noise level (`noise_not_noise` 
        with the highest -1 key) or by the highest silhouette score (`best_value_w/o_penalty`).
        Generates a bar plot showing noise levels with a scatter overlay for silhouette values 
        and cluster counts, each scaled to its own maximum.

        Parameters
        ----------
        top_n : int, optional
            Number of top experiments to display in the plot, default is 20.
        priority : str, optional
            Sorting priority: "noise" to sort first by noise count, "eval_method" to sort first by silhouette score.
            Default is "eval_method".
        show_plots : bool, optional
            If True, displays the plot. Default is False.
        """
        # Check if `results_df` contains data
        if self.experiment.results_df is None or self.experiment.results_df.empty:
            logger.warning("No results found in the experiment DataFrame.")
            return

        # Create temporary columns for sorting and extracting data
        self.experiment.results_df['noise_count'] = self.experiment.results_df["noise_not_noise"].apply(lambda x: x[-1])
        self.experiment.results_df['silhouette_score'] = self.experiment.results_df["best_value_w/o_penalty"]
        self.experiment.results_df['cluster_count'] = self.experiment.results_df["label_counter"].apply(lambda x: len(x) - (1 if -1 in x else 0))

        # Define sorting order based on the priority parameter
        if priority == "noise":
            # Sort first by noise count, then by silhouette score
            sorted_experiments = self.experiment.results_df.sort_values(
                by=["noise_count", "silhouette_score"],
                ascending=[False, False]
            ).head(top_n)
        elif priority == "eval_method":
            # Sort first by silhouette score, then by noise count
            sorted_experiments = self.experiment.results_df.sort_values(
                by=["silhouette_score", "noise_count"],
                ascending=[False, False]
            ).head(top_n)
        else:
            logger.warning("Invalid priority specified. Choose 'noise' or 'eval_method'.")
            return

        # Extract and normalize data for plotting
        noise_counts = sorted_experiments["noise_count"].values
        silhouette_scores = sorted_experiments["silhouette_score"].values
        cluster_counts = sorted_experiments["cluster_count"].values

        # Normalize values for proportional representation
        max_noise = max(noise_counts) if max(noise_counts) > 0 else 1
        max_silhouette = max(silhouette_scores) if max(silhouette_scores) > 0 else 1
        max_clusters = max(cluster_counts) if max(cluster_counts) > 0 else 1

        normalized_noise = noise_counts / max_noise
        normalized_silhouette = silhouette_scores / max_silhouette
        normalized_clusters = cluster_counts / max_clusters

        # Set up the plot
        plt.figure(figsize=(12, 6))

        # Bar plot for normalized noise counts
        plt.bar(range(top_n), normalized_noise, color="skyblue", label="Normalized Noise Count (-1)")

        # Scatter plot for normalized silhouette scores
        plt.scatter(range(top_n), normalized_silhouette, color="orange", label="Normalized Silhouette Score", zorder=5)

        # Scatter plot for normalized cluster counts
        plt.scatter(range(top_n), normalized_clusters, color="green", marker='x', label="Normalized Cluster Count", zorder=6)

        # Labels and title
        plt.xlabel("Experiment Rank (Top Noise-Silhouette)")
        plt.ylabel("Normalized Counts / Scores")
        plt.title("Top Experiments by Noise Level, Silhouette Score, and Cluster Count")
        plt.xticks(range(top_n), labels=range(1, top_n + 1))
        plt.legend(loc="upper right")

        # Save and display the plot
        if priority == "eval_method":
            plt.savefig(self.add_path_type("top_silhouette_noise_clusters"), bbox_inches="tight")
        else:
            plt.savefig(self.add_path_type("top_noise_silhouette_clusters"), bbox_inches="tight")
            
        if show_plots:
            plt.show()

        # Clean up the temporary columns
        self.experiment.results_df.drop(columns=["noise_count", "silhouette_score", "cluster_count"], inplace=True)

        logger.info(f"Top experiments plot generated with priority on {priority}.")




    def show_top_silhouette_noise_ratio(self, top_n=20, show_plots=False):
        """
        Calculates the ratio of silhouette score to noise count (silhouette / (noise + 1)), sorts experiments by this ratio,
        and generates a bar plot showing the silhouette-to-noise ratio with scatter overlays for the silhouette score and noise count.

        Parameters
        ----------
        top_n : int, optional
            Number of top experiments to display in the plot, default is 20.
        show_plots : bool, optional
            If True, displays the plot. Default is False.
        """
        # Check if `results_df` contains data
        if self.experiment.results_df is None or self.experiment.results_df.empty:
            logger.warning("No results found in the experiment DataFrame.")
            return

        # Calculate noise count and silhouette score columns for easy access
        self.experiment.results_df['noise_count'] = self.experiment.results_df["noise_not_noise"].apply(lambda x: x[-1])
        self.experiment.results_df['silhouette_score'] = self.experiment.results_df["best_value_w/o_penalty"]

        # Calculate silhouette-to-noise ratio
        self.experiment.results_df['silhouette_noise_ratio'] = self.experiment.results_df['silhouette_score'] / (self.experiment.results_df['noise_count'] + 1)

        # Sort by silhouette-to-noise ratio in descending order and select the top `top_n`
        sorted_experiments = self.experiment.results_df.sort_values(
            by="silhouette_noise_ratio", ascending=False
        ).head(top_n)

        # Extract data for plotting
        silhouette_scores = sorted_experiments["silhouette_score"].values
        noise_counts = sorted_experiments["noise_count"].values
        silhouette_noise_ratios = sorted_experiments["silhouette_noise_ratio"].values
        experiment_indices = sorted_experiments.index.tolist()  # Get the indices of the top experiments

        # Normalize values for proportional representation
        max_ratio = max(silhouette_noise_ratios) if max(silhouette_noise_ratios) > 0 else 1
        max_silhouette = max(silhouette_scores) if max(silhouette_scores) > 0 else 1
        max_noise = max(noise_counts) if max(noise_counts) > 0 else 1

        normalized_ratios = silhouette_noise_ratios / max_ratio
        normalized_silhouette = silhouette_scores / max_silhouette
        normalized_noise = noise_counts / max_noise

        # Set up the plot
        plt.figure(figsize=(12, 6))

        # Bar plot for the normalized silhouette-to-noise ratios
        plt.bar(range(top_n), normalized_ratios, color="skyblue", label="Normalized Silhouette-to-Noise Ratio")

        # Scatter plot for normalized silhouette scores
        plt.scatter(range(top_n), normalized_silhouette, color="orange", label="Normalized Silhouette Score", zorder=5)

        # Scatter plot for normalized noise counts
        plt.scatter(range(top_n), normalized_noise, color="red", marker='x', label="Normalized Noise Count", zorder=6)

        # Labels and title
        plt.xlabel("Experiment Index (Top Silhouette-to-Noise Ratio)")
        plt.ylabel("Normalized Values")
        plt.title("Top Experiments by Silhouette-to-Noise Ratio with Silhouette Score and Noise Count")
        plt.xticks(range(top_n), labels=experiment_indices, rotation=45)
        plt.legend(loc="upper right")

        # Save and display the plot
        plt.savefig(self.add_path_type("top_silhouette_noise_ratio"), bbox_inches="tight")
        if show_plots:
            plt.show()

        # Clean up the temporary columns
        self.experiment.results_df.drop(columns=["noise_count", "silhouette_score", "silhouette_noise_ratio"], inplace=True)

        logger.info("Top experiments plot generated by silhouette-to-noise ratio.")
            






    
if __name__ == "__main__":
    pass
