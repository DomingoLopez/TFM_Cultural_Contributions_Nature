import seaborn as sns
import pandas as pd
import numpy as np
import pylab as plt
from loguru import logger
import sys

from typing import Optional, Tuple
from matplotlib.colors import ListedColormap





class Clustering:
    """
    A class for performing clustering on datasets using various algorithms.

    This class enables clustering (grouping) of data with multiple models and 
    optionally allows hyperparameter tuning to optimize model performance. It includes
    validation checks to ensure that the specified clustering type and hyperparameters
    are valid and features a logging system for feedback during the process.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to be clustered, provided as a pandas `DataFrame`.
    clustering_type : str
        The clustering algorithm to use. Must be one of the following:
        "kmeans", "hdbscan", "dbscan", "gmm", "tree-bottom-up", "tree-top-down".
    hyperparamtuning : bool, optional
        Specifies whether hyperparameter tuning should be performed for the model. Defaults to `False`.
    hyperparams : dict, optional
        A dictionary containing the hyperparameters specific to the selected clustering algorithm. 
        Each algorithm type accepts a unique set of hyperparameters:
        - "kmeans": `n_clusters`, `init`, `n_init`, `max_iter`
        - "gmm": `n_components`, `covariance_type`, `init_params`, `max_iter`
        - "dbscan": `eps`, `min_samples`, `metric`
        - "tree-bottom-up": `n_clusters`, `linkage`
        - "hdbscan" and "tree-top-down": Placeholder parameters for future implementation.
    verbose : bool, optional
        Enables detailed logging if set to `True`. By default, logging is set to `INFO` level.

    Attributes
    ----------
    accepted_models : tuple
        The list of accepted clustering models.
    accepted_hyperparams : dict
        A dictionary specifying the allowed hyperparameters for each model type.
    data : pd.DataFrame
        The dataset for clustering.
    clustering_type : str
        The chosen clustering model type.
    hiperparamtuning : bool
        Indicates whether hyperparameter tuning will be performed.
    
    Methods
    -------
    __validate_correct_clustering_types()
        Validates if the specified clustering type is among the accepted models.
    __validate_correct_hyperparams()
        Validates if the given hyperparameters align with those accepted by the specified clustering type.
    
    Raises
    ------
    ValueError
        If `clustering_type` is not an accepted model or if invalid hyperparameters are specified.
    TypeError
        If `hyperparams` is not provided as a dictionary.
    """

    def __init__(self, 
                 data: pd.DataFrame, 
                 clustering_type: str, 
                 hyperparamtuning: bool = False,
                 hyperparams: Optional[dict] = None,
                 verbose: bool = False):
        
        # Attrs
        self.accepted_models = ("kmeans", "hdbscan", "dbscan", "gmm", "tree-botton-up", "tree-top-down")
        self.accepted_hyperparams = {
            "kmeans": "n_clusters, init, n_init, max_iter",
            "gmm": "n_components, covariance_type, init_params, max_iter",
            "dbscan": "eps, min_samples, metric",
            "tree-bottom-up": "n_clusters, linkage",
            "hdbscan": "TODO",
            "tree-top-down": "TODO"
        }
        self.data = data
        self.clustering_type = clustering_type
        self.hiperparamtuning = hyperparamtuning

        # Setup logging
        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        # Sanity checks
        if clustering_type not in self.accepted_models:
            raise ValueError(f"Invalid {clustering_type=}. Accepted models are {self.accepted_models}")
        
        if hyperparams != None and not isinstance(hyperparams, dict):
            raise TypeError(f"Expected 'hiperparams' to be a dictionary.")

        if not self.__validate_correct_hyperparams():
            raise ValueError(f"Invalid hyperparameters for given {clustering_type=}. Accepted hyperparams are {self.accepted_hyperparams.get(clustering_type)}")




    def __validate_correct_hyperparams(self):
        """
        Validate that hyperparams are correct depending on 
        clustering type
        """

        if self.clustering_type == "kmeans":
            pass
        elif self.clustering_type == "gmm":
            pass
        elif self.clustering_type == "dbscan":
            pass
        elif self.clustering_type == "hdbscan":
            pass
        elif self.clustering_type == "tree-bottom-up":
            pass
        elif self.clustering_type == "tree-top-down":
            pass
        else:
            pass




    def run_clustering(self):
        """
        Run certain clustering experiment using 
        hyperparam optimizations
        """
        pass



    def show_clustering_plot(
        self,
        X: pd.DataFrame, 
        c: Optional[np.ndarray] = None, 
        centroids: Optional[np.ndarray] = None,
        i: int = 0, 
        j: int = 0, 
        figs: Tuple[int, int] = (9, 7)
    ):
        """
        Plots 2D dataset and its associated clusters.

        Parameters
        ----------
        X : pd.DataFrame
            Data points to plot, with each row representing a sample.
        c : Optional[np.ndarray]
            Cluster labels for each point.
        centroids : Optional[np.ndarray]
            Coordinates of cluster centroids.
        i : int
            Index of the feature for the x-axis.
        j : int
            Index of the feature for the y-axis.
        figs : Tuple[int, int]
            Size of the figure.
        """

        # color mapping for clusters
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#FFFF00', '#0000FF','#FF9D0A','#00B6FF','#F200FF','#FF6100'])
        # Plotting frame
        plt.figure(figs)
        # Plotting points with seaborn
        sns.scatterplot(x=X.iloc[:, i], y=X.iloc[:, j], hue=c, palette=cmap_bold.colors, s=30)
        # Plotting centroids
        if centroids is not None:
            sns.scatterplot(x=centroids[:, i], y=centroids[:, j], marker='D',palette=cmap_bold.colors, hue=range(centroids.shape[0]), s=100,edgecolors='black')
        # Show plot
        plt.show()






if __name__ == "__main__":
    pass