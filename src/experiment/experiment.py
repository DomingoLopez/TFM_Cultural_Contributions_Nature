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
from src.clustering.clustering_model import ClusteringModel
from src.eda.eda import EDA


class Experiment():
    """
    Experiment Class where we can define which kind of methods, algorithms, 
    scalers and optimizers we should experiment with

    This is the main class for doing experiments
    """

    def __init__(self, 
                 data: pd.DataFrame | list, 
                 optimizer: str,
                 dim_reduction: bool, 
                 dim_reduction_range: tuple,
                 scalers: tuple, 
                 clustering: str,
                 eval_method: str,
                 penalty: str,
                 penalty_range: tuple,
                 cache: True, 
                 verbose: False,
                 **kwargs):

        # Setup attrs
        self.data = data
        self.optimizer = optimizer
        self.dim_reduction = dim_reduction
        self.dim_reduction_range = dim_reduction_range
        self.scalers = scalers
        self.clustering = clustering
        self.eval_method = eval_method
        self.penalty = penalty
        self.penalty_range = penalty_range
        self.cache = cache
        # Eda object
        self.eda = EDA(self.data, verbose=False)

        # Setup logging
        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")
        

        # Dirs and files
        self.main_result_dir = Path(__file__).resolve().parent / \
                                f"results/{self.clustering}" / \
                                f"{"optuna" if self.optimizer == "optuna" else "gridsearch"}" 
        self.result_path_csv = os.path.join(self.main_result_dir, f"{self.eval_method}_penalty_{self.penalty}.csv")
        self.result_path_pkl = os.path.join(self.main_result_dir, f"{self.eval_method}_penalty_{self.penalty}.pkl")
        os.makedirs(self.main_result_dir, exist_ok=True)
                                
       



    def run_experiment(self):
        
        logger.info(f"STARTING EXPERIMENT USING {self.optimizer.upper()} OPTIMIZER")
        if self.optimizer == "optuna":
            self.run_experiment_optuna()
        elif self.optimizer == "gridsearch":
            self.run_experiment_gridsearch()
        else:
            raise ValueError("optimizer not supported. Valid options are 'optuna' or 'gridsearch' ")
    



    def run_experiment_optuna(self):

        # If file exists and cache=True
        if os.path.isfile(self.result_path_pkl) and self.cache:
            try:
                results_df = pickle.load(
                    open(str(self.result_path_pkl), "rb")
                )
                #resave as csv
                results_df.to_csv(self.result_path_csv,sep=";")
            except:
                FileNotFoundError("Couldnt find provided file with results from experiment. Please, ensure that file exists.")
        # If no cache or no file found
        else:
            for scaler in self.scalers:
                embeddings_scaled = eda.run_scaler(scaler)
                for dim in range(15, 25):
                    embeddings_after_dimred = eda.run_dim_red(embeddings_scaled, dimensions=dim, dim_reduction=dim_red, show_plots=False)
                    clustering_model = ClusteringFactory.create_clustering_model(clustering, embeddings_after_dimred)
                    # Execute optuna
                    study = clustering_model.run_optuna(evaluation_method=eval_method, n_trials=100, penalty=penalty, penalty_range=penalty_range)
                    # Access best trial n_cluster
                    best_trial = study.best_trial
                    n_clusters_best = best_trial.user_attrs.get("n_clusters", None)  # Extract clusters
                    centers_best = best_trial.user_attrs.get("centers", None)  # Extract centers
                    score_best = best_trial.user_attrs.get("score_original", None)  # Extract original score
                    # Store results
                    results.append({
                        "scaler": scaler,
                        "dim_reduction":dim_red,
                        "dimension": dim,
                        "n_clusters": n_clusters_best,
                        "best_params": str(study.best_params),
                        "centers": centers_best,
                        "best_value": study.best_value,
                        "best_original_value": score_best
                    })

            # Store results as dataframe and csv in cache
            results_df = pd.DataFrame(results)
            results_df.to_csv(result_file_cache_path_csv,sep=";")
            # Save study as object in cache.
            results_cache_path = ""
            pickle.dump(
                results_df,
                open(str(result_file_cache_path), "wb")
            )
       







if __name__ == "__main__":
    pass
