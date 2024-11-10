from collections import Counter
import os
from pathlib import Path
import pickle
import sys
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_factory import ClusteringFactory
from src.clustering.clustering_model import ClusteringModel
from src.eda.eda import EDA


class Experiment():
    """
    Experiment Class where we can define which kind of methods, algorithms, 
    scalers and optimizers we should experiment with.
    
    This is the main class for setting up and running clustering experiments 
    using specified optimizers, dimensionality reduction methods, and evaluation metrics.
    """

    def __init__(self, 
                 data: list, 
                 optimizer: str,
                 dim_reduction: bool, 
                 dim_reduction_range: list,
                 scalers: list, 
                 clustering: str,
                 eval_method: str,
                 penalty: str,
                 penalty_range: tuple,
                 cache:bool= True, 
                 verbose:bool= False,
                 **kwargs):
        """
        Initializes an experiment with the specified configuration.

        Args:
            data (list): The data to be used for the experiment.
            optimizer (str): The optimization method to use, e.g., 'optuna' or 'gridsearch'.
            dim_reduction (bool): Whether to apply dimensionality reduction.
            dim_reduction_range (list): Range of dimensions to reduce the data to.
            scalers (list): List of scalers to normalize the data.
            clustering (str): Clustering algorithm to apply.
            eval_method (str): Evaluation metric for clustering quality.
            penalty (str): Penalty type to be applied in optimization.
            penalty_range (tuple): Range of penalty values.
            cache (bool): If True, caching is enabled.
            verbose (bool): If True, enables verbose logging.
            **kwargs: Additional keyword arguments.
        """
        # Setup attrs
        self._data = data
        self._optimizer = optimizer
        self._dim_reduction = dim_reduction
        self._dim_reduction_range = dim_reduction_range
        self._scalers = scalers
        self._clustering = clustering
        self._eval_method = eval_method
        self._penalty = penalty
        self._penalty_range = penalty_range
        self._cache = cache
        self._verbose = verbose
        self._eda = EDA(self._data, verbose=False, cache=self._cache)
        self._results_df = None

        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        self._main_result_dir = (
            Path(__file__).resolve().parent
            / f"results/{self._clustering}"
            / ("optuna" if self._optimizer == "optuna" else "gridsearch")
            / f"dim_red_{self._dim_reduction}"
        )
        self._result_path_csv = os.path.join(self._main_result_dir, f"{self._eval_method}_penalty_{self._penalty}.csv")
        self._result_path_pkl = os.path.join(self._main_result_dir, f"{self._eval_method}_penalty_{self._penalty}.pkl")
        os.makedirs(self._main_result_dir, exist_ok=True)
                                
    # Getters and Setters
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def eda(self):
        return self._eda

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def dim_reduction(self):
        return self._dim_reduction

    @dim_reduction.setter
    def dim_reduction(self, value):
        self._dim_reduction = value

    @property
    def dim_reduction_range(self):
        return self._dim_reduction_range

    @dim_reduction_range.setter
    def dim_reduction_range(self, value):
        self._dim_reduction_range = value

    @property
    def scalers(self):
        return self._scalers

    @scalers.setter
    def scalers(self, value):
        self._scalers = value

    @property
    def clustering(self):
        return self._clustering

    @clustering.setter
    def clustering(self, value):
        self._clustering = value

    @property
    def eval_method(self):
        return self._eval_method

    @eval_method.setter
    def eval_method(self, value):
        self._eval_method = value

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        self._penalty = value

    @property
    def penalty_range(self):
        return self._penalty_range

    @penalty_range.setter
    def penalty_range(self, value):
        self._penalty_range = value

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    @property
    def results_df(self):
        return self._results_df

    @results_df.setter
    def results_df(self, value):
        self._results_df = value
    
       

    def run_experiment(self):
        """
        Executes the experiment based on the chosen optimizer.
        
        Calls the appropriate internal method for running an experiment using 
        either Optuna or Grid Search as specified in the optimizer attribute.
        
        Raises:
            ValueError: If the optimizer specified is not supported.
        """
        logger.info(f"STARTING EXPERIMENT USING {self._optimizer.upper()} OPTIMIZER")
        if self._optimizer == "optuna":
            self.__run_experiment_optuna()
        elif self._optimizer == "gridsearch":
            self.__run_experiment_gridsearch()
        else:
            raise ValueError("optimizer not supported. Valid options are 'optuna' or 'gridsearch' ")
    


    def __run_experiment_optuna(self):
        """
        Runs the experiment using the Optuna optimizer.
        
        If cache is enabled and results exist, it loads them from a pickle file.
        Otherwise, it performs the optimization and saves results to CSV and pickle.
        
        Raises:
            FileNotFoundError: If the cached results file is not found.
        """
        # If file exists and cache=True
        if os.path.isfile(self._result_path_pkl) and self._cache:
            try:
                results_df = pickle.load(open(str(self._result_path_pkl), "rb"))
                results_df.to_csv(self._result_path_csv, sep=";")
                self._results_df = results_df
            except FileNotFoundError:
                raise FileNotFoundError("Couldn't find provided file with results from experiment. Please ensure that file exists.")
        else:
            results = []
            for scaler in self._scalers:
                embeddings_scaled = self._eda.run_scaler(scaler)
                for dim in range(self._dim_reduction_range[0], self._dim_reduction_range[1], 1):
                    embeddings_after_dimred = self._eda.run_dim_red(
                        embeddings_scaled, dimensions=dim, dim_reduction=self._dim_reduction, scaler=scaler, show_plots=False
                    )
                    clustering_model = ClusteringFactory.create_clustering_model(self._clustering, embeddings_after_dimred)
                    study = clustering_model.run_optuna(
                        evaluation_method=self._eval_method, n_trials=100, penalty=self._penalty, penalty_range=self._penalty_range
                    )
                    best_trial = study.best_trial
                    n_clusters_best = best_trial.user_attrs.get("n_clusters", None)
                    centers_best = best_trial.user_attrs.get("centers", None)
                    labels_best = best_trial.user_attrs.get("labels", None)
                    label_counter = Counter(labels_best)
                    score_best = best_trial.user_attrs.get("score_original", None)

                    noise_not_noise = {
                        -1: label_counter.get(-1, 0),
                        1: sum(v for k, v in label_counter.items() if k != -1)
                    }

                    silhouette_noise_ratio = score_best / (noise_not_noise.get(-1) + 1)
                    
                    results.append({
                        "clustering": self._clustering,
                        "optimization": self._optimizer,
                        "scaler": scaler,
                        "dim_reduction": self._dim_reduction,
                        "dimensions": dim,
                        "embeddings": embeddings_after_dimred,
                        "n_clusters": n_clusters_best,
                        "best_params": str(study.best_params),
                        "centers": centers_best,
                        "labels": labels_best,
                        "label_counter": label_counter,
                        "noise_not_noise": noise_not_noise,
                        "silhouette_noise_ratio": silhouette_noise_ratio,
                        "penalty": self._penalty,
                        "penalty_range": self._penalty_range if self._penalty is not None else None,
                        "best_value_w_penalty": study.best_value,
                        "best_value_w/o_penalty": score_best
                    })
            logger.info(f"ENDING EXPERIMENT...STORING RESULTS.")
            results_df = pd.DataFrame(results)
            results_df.to_csv(self._result_path_csv, sep=";")
            self._results_df = results_df
            pickle.dump(results_df, open(str(self._result_path_pkl), "wb"))
            logger.info(f"EXPERIMENT ENDED.")
       

    def __run_experiment_gridsearch(self):
        """
        Runs the experiment using Grid Search.

        If cache is enabled and results exist, it loads them from a pickle file.
        Otherwise, it performs the grid search and saves results to CSV and pickle.
        
        Raises:
            FileNotFoundError: If the cached results file is not found.
        """
        # If file exists and cache=True
        if os.path.isfile(self._result_path_pkl) and self._cache:
            try:
                results_df = pickle.load(open(str(self._result_path_pkl), "rb"))
                # Resave as CSV
                results_df.to_csv(self._result_path_csv, sep=";")
                # Update results
                self._results_df = results_df
            except FileNotFoundError:
                raise FileNotFoundError("Couldn't find provided file with results from experiment. Please ensure that file exists.")
        else:
            results = []
            for scaler in self._scalers:
                embeddings_scaled = self._eda.run_scaler(scaler)
                for dim in range(self._dim_reduction_range[0], self._dim_reduction_range[1], 1):
                    embeddings_after_dimred = self._eda.run_dim_red(
                        embeddings_scaled, dimensions=dim, dim_reduction=self._dim_reduction, show_plots=False
                    )
                    clustering_model = ClusteringFactory.create_clustering_model(self._clustering, embeddings_after_dimred)
                    # Execute Grid Search
                    grid_search = clustering_model.run_gridsearch(evaluation_method=self._eval_method)

                    # Iterate over all grid search results
                    for i in range(len(grid_search.cv_results_['params'])):
                        params = grid_search.cv_results_['params'][i]
                        score = grid_search.cv_results_['mean_test_score'][i]

                        # Get n_clusters from params or estimate it from labels if available
                        n_clusters = params.get("n_clusters", None)
                        estimator = grid_search.estimator.set_params(**params).fit(embeddings_after_dimred)
                        labels = getattr(estimator, 'labels_', None)
                        if n_clusters is None and labels is not None:
                            # Count clusters excluding noise (-1)
                            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                        centers = clustering_model.get_cluster_centers(labels)
                        label_counter = Counter(labels)

                        noise_not_noise = {
                            -1: label_counter.get(-1, 0),
                            1: sum(v for k, v in label_counter.items() if k != -1)
                        }

                        silhouette_noise_ratio = score / (noise_not_noise.get(-1) + 1)

                        results.append({
                            "clustering": self._clustering,
                            "optimization": self._optimizer,
                            "scaler": scaler,
                            "dim_reduction": self._dim_reduction,
                            "dimensions": dim,
                            "embeddings": embeddings_after_dimred,
                            "n_clusters": n_clusters,
                            "params": str(params),
                            "centers": centers,
                            "labels": labels,
                            "label_counter": label_counter,
                            "noise_not_noise": noise_not_noise,
                            "silhouette_noise_ratio": silhouette_noise_ratio,
                            "penalty": None,
                            "penalty_range": None,
                            "value_w_penalty": None,
                            "value_w/o_penalty": score
                        })

            logger.info(f"ENDING EXPERIMENT...STORING RESULTS.")
            # Save results as DataFrame and to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(self._result_path_csv, sep=";")
            # Update results
            self._results_df = results_df
            # Save results as pickle
            pickle.dump(results_df, open(self._result_path_pkl, "wb"))
            logger.info("Results saved.")
            logger.info(f"EXPERIMENT ENDED.")



if __name__ == "__main__":
    pass
