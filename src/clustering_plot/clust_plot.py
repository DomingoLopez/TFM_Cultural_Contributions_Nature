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
from src.eda.eda import EDA
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
        

    
    def show_best_silhouette(self, top_n=15, min_clusters=30, show_plots=False):
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

        # Filter to get the experiment with the best silhouette score
        best_experiment = self.experiment.results_df.loc[self.experiment.results_df['best_value_w/o_penalty'].idxmax()]

        # Extract information for the best configuration
        best_labels = best_experiment['labels']
        scaler = best_experiment['scaler']
        dim_reduction = best_experiment['dim_reduction']
        dimensions = best_experiment['dimensions']
        params = best_experiment['best_params']
        optimizer = best_experiment['optimization']
        
        # Get scaled and reduced data for the best configuration
        scaled_data = self.experiment._eda.run_scaler(scaler)
        reduced_data = self.experiment._eda.run_dim_red(
            scaled_data, dimensions=dimensions, dim_reduction=dim_reduction, show_plots=False
        )

        # Calculate silhouette values for each point
        silhouette_values = silhouette_samples(reduced_data, best_labels)
        silhouette_avg = silhouette_score(reduced_data, best_labels)

        # Calculate average silhouette per cluster
        unique_labels = np.unique(best_labels)
        cluster_count = len(unique_labels)

        # If there are `min_clusters` or fewer clusters, plot all clusters
        if cluster_count <= min_clusters:
            logger.info(f"The number of clusters ({cluster_count}) is less than or equal to {min_clusters}. "
                        "All clusters will be plotted.")
            selected_clusters = unique_labels
        else:
            # Compute the average silhouette for each cluster
            cluster_silhouette_means = {
                label: silhouette_values[best_labels == label].mean() for label in unique_labels
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
            ith_cluster_silhouette_values = silhouette_values[best_labels == label]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
            y_lower = y_upper + 10

        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster Index")
        plt.title(f"Silhouette Coefficient Plot for Best Configuration {optimizer}\n\n"
                f"Clustering: {self.experiment.clustering} - Dimensionality Reduction: {self.experiment.dim_reduction}\n"
                f"Params: {params}"
                )

        # Save the plot
        plt.savefig(self.add_path_type("best_trial_silhouette"), bbox_inches='tight')
        if show_plots:
            plt.show()

        logger.info("Silhouette plot generated for the selected clusters.")

    
    
    
    
    def show_best_scatter(self, show_plots=False):
        """
        Plots a 2D scatter plot for the best experiment configuration, with clusters reduced 
        to 2D space using PCA and color-coded for better visual distinction. Points labeled 
        as noise (-1) are always shown in red.
        """
        
        # Convert data to numpy array if it's a list
        data = np.array(self.experiment.data) if isinstance(self.experiment.data, list) else self.experiment.data
        
        # Get best experiment data
        best_experiment = self.experiment.results_df.loc[self.experiment.results_df['best_value_w/o_penalty'].idxmax()]
        best_labels = np.array(best_experiment['labels'])
        cluster_count = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)  # Exclude noise (-1) from cluster count

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
        
        plt.title(f"Scatter Plot of Best Experiment (Noise in Red, Clusters in 2D PCA)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # Save and show plot
        plt.savefig(self.add_path_type("best_experiment_scatter"), bbox_inches='tight')
        if show_plots:
            plt.show()
        
    
    
    
    
    
    def show_best_scatter_with_centers(self, show_plots=False):
        """
        Plots a 2D scatter plot for the best experiment configuration, with clusters reduced 
        to 2D space using PCA and color-coded for better visual distinction. Points labeled 
        as noise (-1) are always shown in red.
        """
        
        # Get best experiment data
        best_experiment = self.experiment.results_df.loc[self.experiment.results_df['best_value_w/o_penalty'].idxmax()]
        # TODO Need 
        best_labels = np.array(best_experiment['labels'])
        # Convert embeddings and centers to numpy arrays if they are DataFrames
        data = best_experiment['embeddings'].values if isinstance(best_experiment['embeddings'], pd.DataFrame) else np.array(best_experiment['embeddings'])
        best_centers = best_experiment['centers'].values if isinstance(best_experiment['centers'], pd.DataFrame) else np.array(best_experiment['centers'])
        cluster_count = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)  # Exclude noise (-1) from cluster count

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

        
        plt.title("Scatter Plot of Best Experiment with Cluster Centers (Noise in Red)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # Save and show plot
        plt.savefig(self.add_path_type("best_experiment_scatter_with_centers"), bbox_inches='tight')
        if show_plots:
            plt.show()

                           
    # def scatter_plot(
    #     self,
    #     X,
    #     labels: Optional[np.ndarray] = None, 
    #     centers: Optional[np.ndarray] = None,
    #     i: int = 0, 
    #     j: int = 0, 
    #     figs: Tuple[int, int] = (9, 7)
    # ):
    #     """
    #     Plots a 2D representation of the dataset and its associated clusters.

    #     This method saves a plot showing the clustering of the 2D-reduced data points,
    #     optionally marking the cluster centroids if provided.

    #     Parameters
    #     ----------
    #     c : Optional[np.ndarray]
    #         Cluster labels for each point.
    #     centroids : Optional[np.ndarray]
    #         Coordinates of cluster centroids in 2D space.
    #     i : int
    #         Index of the feature for the x-axis.
    #     j : int
    #         Index of the feature for the y-axis.
    #     figs : Tuple[int, int]
    #         Size of the figure in inches.
    #     save_path : str
    #         Path to store the plot image.
    #     """

    #     X = self.experiment.data

    #     # color mapping for clusters
    #     colors = ['#FF0000', '#00FF00', '#FFFF00', '#0000FF', '#FF9D0A', '#00B6FF', '#F200FF', '#FF6100']
    #     cmap_bold = ListedColormap(colors)
    #     # Plotting frame
    #     plt.figure(figsize=figs)
    #     # Plotting points with seaborn
    #     sns.scatterplot(x=X.iloc[:, i], y=X.iloc[:, j], hue=labels, palette=cmap_bold.colors, s=30, hue_order=sorted(set(labels)))  # Ensures that -1 appears first in the legend if present)
    #     # Plotting centroids
    #     if centers is not None:
    #         sns.scatterplot(x=centers[:, i], y=centers[:, j], marker='D',palette=colors[1:] if -1 in set(labels) else colors[:], hue=range(centers.shape[0]), s=100,edgecolors='black')
    #     # Save plot making 
    #     plt.savefig(self.replace_path_type("scatter"), bbox_inches='tight')
    
    
    
    
    
    # def do_PCA_for_representation(self, df, centers):
    #     """
    #     Performs PCA to reduce data and cluster centers to 2D for plotting.

    #     This function applies PCA (Principal Component Analysis) to the data and 
    #     optionally to the cluster centers, reducing them to 2D space for visualization 
    #     purposes.

    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         Data points to reduce, where each row represents a sample.
    #     centers : np.ndarray
    #         Coordinates of cluster centroids before dimensionality reduction.

    #     Returns
    #     -------
    #     pca_df : pd.DataFrame
    #         Data points reduced to 2D space.
    #     pca_centers : np.ndarray
    #         Cluster centroids reduced to 2D space.
    #     """
    #     if df.shape[1] > 2:
    #         pca = PCA(n_components=2,random_state=42)
    #         pca_result = pca.fit_transform(df.values)
    #         pca_df = pd.DataFrame(data=pca_result)
    #         pca_centers = pca.transform(centers)
    #         return pca_df, pca_centers
    #     else:
    #         return df, centers

    
    
if __name__ == "__main__":
    pass
