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
    scalers and optimizers we should experiment with

    This is the main class for doing experiments
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
        self.main_result_dir = Path(__file__).resolve().parent.parent.parent / \
                                f"results/{self.clustering}" / \
                                "optuna" if self.optimizer == "optuna" else "gridsearch" / \
                                f"dim_red_{self.dim_reduction}" 
        self.result_path_csv = os.path.join(self.main_result_dir, f"{self.eval_method}_penalty_{self.penalty}.csv")
        self.result_path_pkl = os.path.join(self.main_result_dir, f"{self.eval_method}_penalty_{self.penalty}.pkl")
        os.makedirs(self.main_result_dir, exist_ok=True)
                                
       

    def run_experiment(self):
        
        logger.info(f"STARTING EXPERIMENT USING {self.optimizer.upper()} OPTIMIZER")
        if self.optimizer == "optuna":
            self.__run_experiment_optuna()
        elif self.optimizer == "gridsearch":
            self.__run_experiment_gridsearch()
        else:
            raise ValueError("optimizer not supported. Valid options are 'optuna' or 'gridsearch' ")
    


    def __run_experiment_optuna(self):

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
            # results var
            results = []
            for scaler in self.scalers:
                embeddings_scaled = self.eda.run_scaler(scaler)
                for dim in self.dim_reduction_range:
                    embeddings_after_dimred = self.eda.run_dim_red(embeddings_scaled, dimensions=dim, dim_reduction=self.dim_reduction, show_plots=False)
                    clustering_model = ClusteringFactory.create_clustering_model(self.clustering, embeddings_after_dimred)
                    # Execute optuna
                    study = clustering_model.run_optuna(evaluation_method=self.eval_method, n_trials=100, penalty=self.penalty, penalty_range=self.penalty_range)
                    # Access best trial n_cluster
                    best_trial = study.best_trial
                    n_clusters_best = best_trial.user_attrs.get("n_clusters", None)  # Extract clusters
                    centers_best = best_trial.user_attrs.get("centers", None)  # Extract centers
                    score_best = best_trial.user_attrs.get("score_original", None)  # Extract original score
                    # Store results
                    results.append({
                        "optimization": self.optimizer,
                        "scaler": scaler,
                        "dim_reduction":self.dim_reduction,
                        "dimensions": dim,
                        "n_clusters": n_clusters_best,
                        "best_params": str(study.best_params),
                        "centers": centers_best,
                        "penalty": self.penalty,
                        "penalty_range": self.penalty_range if self.penalty is not None else None,
                        "best_value_w_penalty": study.best_value,
                        "best_value_w/o_penalty": score_best
                    })
            logger.info(f"ENDING EXPERIMENT...STORING RESULTS.")
            # Store results as dataframe and csv in result folder
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.result_path_csv,sep=";")
            # Save study as object.
            pickle.dump(
                results_df,
                open(str(self.result_path_pkl), "wb")
            )
            logger.info(f"EXPERIMENT ENDED.")
       

    def __run_experiment_gridsearch(self):

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
            results = []
            for scaler in self.scalers:
                embeddings_scaled = self.eda.run_scaler(scaler)
                for dim in self.dim_reduction_range:
                    embeddings_after_dimred = self.eda.run_dim_red(embeddings_scaled, dimensions=dim, dim_reduction=self.dim_reduction, show_plots=False)
                    clustering_model = ClusteringFactory.create_clustering_model(self.clustering, embeddings_after_dimred)
                    # Execute gridSearch
                    grid_search = clustering_model.run_gridsearch(evaluation_method=self.eval_method)
                    # Get Best Results
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                    
                    # Get n clusters
                    n_clusters_best = best_params.get("n_clusters", None)

                    # If no number of clusters provided
                    if n_clusters_best is None and hasattr(grid_search.best_estimator_, 'labels_'):
                        # Get number of clusters avoiding noise (-1)
                        n_clusters_best = len(set(grid_search.best_estimator_.labels_)) - (1 if -1 in grid_search.best_estimator_.labels_ else 0)

                    centers_best = clustering_model.get_cluster_centers(grid_search.best_estimator_.labels_)

                    results.append({
                        "optimization": self.optimizer,
                        "scaler": scaler,
                        "dim_reduction": self.dim_reduction,
                        "dimensions": dim,
                        "n_clusters": n_clusters_best,
                        "best_params": str(best_params),
                        "centers": centers_best,
                        "penalty": None,
                        "penalty_range": None,
                        "best_value_w_penalty":None,
                        "best_value_w/o_penalty": best_score
                    })

            logger.info(f"ENDING EXPERIMENT...STORING RESULTS.")
            # Guardar resultados
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.result_path_csv, sep=";")
            pickle.dump(results_df, open(self.result_path_pkl, "wb"))
            logger.info("Resultados guardados.")
            logger.info(f"EXPERIMENT ENDED.")




if __name__ == "__main__":
    pass
