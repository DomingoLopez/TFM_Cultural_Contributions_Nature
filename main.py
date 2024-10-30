import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from src.clustering.clust_hdbscan import HDBSCANClustering
from src.clustering.clustering_factory import ClusteringFactory
from src.utils.image_loader import ImageLoader
from src.dinov2_inference.dinov2_inference import Dinov2Inference
from src.eda.eda import EDA



if __name__ == "__main__":
    # Finding images
    # image_loader = ImageLoader(folder="./data/Small_Data")
    image_loader = ImageLoader(folder="./data/Small_Data")
    images = image_loader.find_images()
    # Loading images and getting embeddings
    dinomodel = Dinov2Inference(model_name="small", images=images)
    embeddings = dinomodel.run()
    # Create Eda object and apply or not dim reduction
    eda = EDA(embeddings=embeddings, verbose=False)

    # ##############################################################
    # BIG STUDY
    # ##############################################################

    # Variables initialization
    scalers = ["standard","minmax","robust","maxabs"]
    dim_red = "umap"
    clustering = "hdbscan"
    eval_method = "davies_bouldin"
    penalty = "proportional" # linear
    penalty_range = (4,8)
    cache = False
    result_dir_cache_path = Path(__file__).resolve().parent / f"cache/results/{clustering}_{eval_method}_penalty_{penalty}_images_{len(images)}"
    os.makedirs(result_dir_cache_path, exist_ok=True)
    result_file_cache_path = Path(__file__).resolve().parent / result_dir_cache_path / "result.pkl"
    result_file_cache_path_csv = Path(__file__).resolve().parent / result_dir_cache_path / "result.csv"
    results = []

    # If file with results doesnt exists
    if not os.path.isfile(result_file_cache_path) or not cache:
        for scaler in scalers:
            embeddings_scaled = eda.run_scaler(scaler)
            for dim in range(3, 15):
                embeddings_after_dimred = eda.run_dim_red(embeddings_scaled, dimensions=dim, dim_reduction=dim_red, show_plots=False)
                clustering_model = ClusteringFactory.create_clustering_model(clustering, embeddings_after_dimred)
                # Execute optuna
                study = clustering_model.run_optuna(evaluation_method=eval_method, n_trials=50, penalty=penalty, penalty_range=penalty_range)
                # Access best trial n_cluster
                best_trial = study.best_trial
                n_clusters_best = best_trial.user_attrs.get("n_clusters", None)  # Extrae el número de clústeres
                score_best = best_trial.user_attrs.get("score_original", None)  # Extrae el score original sin penalizar por clusteres bajos
                # Store results
                results.append({
                    "scaler": scaler,
                    "dimension": dim,
                    "n_clusters": n_clusters_best,
                    "best_params": str(study.best_params),
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
            open(str(result_file_cache_path), "wb"),
        )
    else:
        try:
            results_df = pickle.load(
                open(str(result_file_cache_path), "rb")
            )
            results_df.to_csv(result_file_cache_path_csv,sep=";")
        except:
            FileNotFoundError("Couldnt find provided file with results from experiments. Please, ensure that file exists.")

    # With results we can make final experiment with desired params
    # For example give me experiment where it got 4 clusters with best
    # value of its metric (davies or silhouette)
    # For now, we could take best result
    best_experiment = results_df.loc[results_df['best_original_value'].idxmin()]
    # clustering_model.run_single_experiment()
    

        
    # # # Create clustering factory and kmeans
    # # # TODO: Here we could pass a eda object to Clustering creation, so it would know how many dimensiones
    # # # do we have and put that in another subfolder with results, or even add that to path name of results.
    # # clustering_model = ClusteringFactory.create_clustering_model("agglomerative", embeddings_after_dimred)
    # # # Run Clustering
    # # study = clustering_model.run_optuna(evaluation_method="silhouette", n_trials=500)
    # # # print("Clustering complete. Results available in results/modelname/timestamp")
