import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from loguru import logger

from src.clustering.clust_hdbscan import HDBSCANClustering
from src.clustering.clustering_factory import ClusteringFactory
from src.experiment.experiment import Experiment
from src.experiment.experiment_result_controller import ExperimentResultController
from src.llava_inference.llava_inference import LlavaInference
from src.utils.image_loader import ImageLoader
from src.dinov2_inference.dinov2_inference import Dinov2Inference
from src.preprocess.preprocess import Preprocess

import matplotlib.pyplot as plt
import cv2



# REFACTOR THIS TO ENSURE IT RECEIVES A DICT AND ONLY TAKE A SAMPLE OF CLUSTERS (150 clusters not viable to show images)
# def show_images_per_cluster(images, knn_cluster_result_df):
#     n_clusters = knn_cluster_result_df.shape[1]  # Número de clusters
#     n_images_per_cluster = knn_cluster_result_df.shape[0]  # Número de imágenes por cluster
    
#     # Crear una figura con subplots para cada cluster y sus imágenes correspondientes
#     fig, axs = plt.subplots(n_clusters, n_images_per_cluster, figsize=(15, n_clusters * 3), squeeze=False)
#     fig.suptitle("Closest Images to Cluster Centers", fontsize=16)
    
#     for cluster_idx in range(n_clusters):
#         cluster_name = f"Cluster_{cluster_idx}"
#         for img_idx in range(n_images_per_cluster):
#             image_index = knn_cluster_result_df[cluster_name].iloc[img_idx]  # Índice de la imagen en el array 'images'
#             img_path = images[image_index]
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para matplotlib
            
#             # Asegurarse de que `axs` siempre sea bidimensional
#             ax = axs[cluster_idx, img_idx]
#             ax.imshow(img)
#             ax.axis("off")
#             ax.set_title(f"{cluster_name} - Img {img_idx + 1}")
    
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar el layout para el título
#     plt.show()




def load_images(path) -> list:
    # Finding images
    # image_loader = ImageLoader(folder="./data/Small_Data")
    image_loader = ImageLoader(folder=path)
    images = image_loader.find_images()
    return images

def generate_embeddings(images, model) -> list:
    # Loading images and getting embeddings
    dinomodel = Dinov2Inference(model_name=model, images=images, disable_cache=False)
    embeddings = dinomodel.run()
    return embeddings


def run_experiments(file, embeddings) -> None:
   
    # Load json file with all experiments
    with open(file, 'r') as f:
        experiments_config = json.load(f)

    for config in experiments_config:
        id = config.get("id")
        optimizer = config.get("optimizer", "optuna")
        optuna_trials = config.get("optuna_trials", None)
        normalization = config.get("normalization", True)
        scaler = config.get("scaler", None)
        dim_red = config.get("dim_red", None)
        reduction_parameters = config.get("reduction_parameters", None)
        clustering = config.get("clustering", "hdbscan")
        eval_method = config.get("eval_method", "silhouette")
        penalty = config.get("penalty", None)
        penalty_range = config.get("penalty_range", None)
        cache = config.get("cache", True)
        # Make and Run Experiment
        logger.info(f"LOADING EXPERIMENT: {id}")
        experiment = Experiment(
            id,
            embeddings,
            optimizer,
            optuna_trials,
            normalization,
            dim_red,
            reduction_parameters,
            scaler,
            clustering,
            eval_method,
            penalty,
            penalty_range,
            cache
        )
        experiment.run_experiment()

# Copy files to ngpu
# rsync -av --exclude='.git' 1_TFM xxxx.xx.es:/mnt/homeGPU/dlopez


if __name__ == "__main__": 
    
    # 1. Load images, generate embeddings and run experiments
    images = load_images("./data/Data")
    embeddings = generate_embeddings(images, model="base")
    experiments_file = "src/experiment/json/experiments_optuna_silhouette_umap.json"
    # experiments_file = "src/experiment/json/single_experiment.json"
    # run_experiments(experiments_file, embeddings)
    #run_experiments("src/experiment/json/experiments_optuna_silhouette_umap.json", embeddings)
    
    # 2. Load all available experiments from results folder
    # 2.1 Define eval method to analyze
    # 2.2 Load all experiments of given eval method
    

    # ###################################################################
    # CLUSTER DIRS AND LLAVA METRIC LOOP

    # 1. Desired filters. Important to select right experiments to consider
    use_score_noise_ratio = False
    # The are range (from 2 to 15)
    reduction_params = {
        "n_components": (2,25),
        "n_neighbors": (3,60),
        "min_dist": (0.1, 0.8)
    }
    n_cluster_range = (60,300)
    # Load json file with all experiments
    with open(experiments_file, 'r') as f:
        experiments_config = json.load(f)

    for config in experiments_config:
        eval_method = config.get("eval_method", "silhouette")
        id = config.get("id",1)
        experiment_controller = ExperimentResultController(eval_method, 
                                                        experiment_id=id, 
                                                        use_score_noise_ratio=use_score_noise_ratio,
                                                        n_cluster_range=n_cluster_range,
                                                        reduction_params=reduction_params
                                                        )
        experiments_filtered = experiment_controller.get_top_k_experiments(top_k=5)
        best_experiment = experiment_controller.get_best_experiment_data(experiments_filtered)
        experiment_controller.plot_all(best_experiment)
        experiment_controller.create_cluster_dirs(images=images, experiment=best_experiment)

        # 3. Process images to Llava-1.5 and see:
        # 3.1 Generate dir with images per cluster (each dir index/name of cluster) - Noise y dir called -1
        #llava = LlavaInference(images=images, classification_lvl=3, n_prompt=1, type="llava")

        # for i in range(1,3,1):
        #     for type in ("llava","llava_next"):
        #         llava = LlavaInference(images=images, classification_lvl=3, best_experiment=best_experiment, n_prompt=i, type=type)
        #         llava.create_cluster_dirs()
        #         # llava.run()
        #         # llava.create_results_stats()
        #         # llava.plot_cluster_categories()


        # I need some actor that allows me to get llava stats from every experiment
        # I got the experiment, I got the results from llava
   
