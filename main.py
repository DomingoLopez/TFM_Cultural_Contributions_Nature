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
from src.multimodal_clustering_metric.multimodal_clustering_metric import MultiModalClusteringMetric
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


def run_experiments(file, images) -> None:
   
    # Load json file with all experiments
    with open(file, 'r') as f:
        experiments_config = json.load(f)

    for config in experiments_config:
        id = config.get("id")
        dino_model = config.get("dino_model","small")
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

        # Generate embeddings based on experiment model
        embeddings = generate_embeddings(images, model=dino_model)
        experiment = Experiment(
            id,
            dino_model,
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
    
    # ###################################################################
    images = load_images("./data/Data")
    #experiments_file = "src/experiment/json/experiments_optuna_umap.json"
    experiments_file = "src/experiment/json/experiments_optuna_tsne.json"
    run_experiments(experiments_file, images)
    experiments_file = "src/experiment/json/experiments_optuna_pca.json"
    run_experiments(experiments_file, images)
    
    # Classification level to analyze
    classification_lvl = [3]
    prompts = [1,2]
    llava_models = ("llava1-5_7b", "llava1-6_7b", "llava1-6_13b")
    # Cluster Range to filter
    n_cluster_range = (40,300)


    # Obtain experiments results
    with open(experiments_file, 'r') as f:
        experiments_config = json.load(f)

    result_list = []
    for config in experiments_config:
        eval_method = config.get("eval_method", "silhouette")
        id = config.get("id",1)
        dino_model = config.get("dino_model")
        dim_red = config.get("dim_red","umap")

        # APPLY FILTERS FROM REDUCTION HIPERPARAMS
        if dim_red == "umap":
            reduction_params = {
                "n_components": (2,25),
                "n_neighbors": (3,60),
                "min_dist": (0.1, 0.8)
            }
        elif dim_red == "tsne":
            reduction_params = {
                "n_components": (2,25),
                "perplexity": (4,60),
                "early_exaggeration": (8, 15)
            }
        else:
            reduction_params = {
                "n_components": (2,25)
            }

        experiment_controller = ExperimentResultController(eval_method, 
                                                           dino_model,
                                                           experiment_id=id, 
                                                           n_cluster_range=n_cluster_range,
                                                           reduction_params=reduction_params)
        experiments_filtered = experiment_controller.get_top_k_experiments(top_k=5)
        best_experiment = experiment_controller.get_best_experiment_data(experiments_filtered)
        experiment_controller.plot_all(best_experiment)
        experiment_controller.create_cluster_dirs(images=images, experiment=best_experiment)

        for class_lvl in classification_lvl:
            for model in llava_models:
                for prompt in prompts:
                    llava = LlavaInference(images=images, classification_lvl=class_lvl, n_prompt=prompt, model=model)
                    llava.run()
                    # Get Llava Results from llava-model i 
                    llava_results_df = llava.get_results(model)
                    # Get cluster of images
                    img_cluster_dict = experiment_controller.cluster_images_dict
                    # Obtain categories from classification_lvl
                    categories = llava.get_categories(class_lvl)
                    # Quality metrics
                    lvm_lvlm_metric = MultiModalClusteringMetric(class_lvl,
                                                                categories,
                                                                model, 
                                                                prompt, 
                                                                best_experiment, 
                                                                img_cluster_dict, 
                                                                llava_results_df)
                    lvm_lvlm_metric.generate_stats()
                    # Obtain results
                    quality_results = pd.DataFrame()
                    for i in (True, False):
                        # Calculate metrics
                        results = lvm_lvlm_metric.calculate_clustering_quality(use_noise=i)
                        # Join results (in columns)
                        quality_results = pd.concat([quality_results, pd.DataFrame([results])], axis=1)

                    # Save results in list
                    result_list.append({
                        "experiment_id" : id,
                        "best_experiment_index": best_experiment["original_index"],
                        "dino_model" : dino_model,
                        "normalization" : best_experiment["normalization"],
                        "scaler" : best_experiment["scaler"],
                        "dim_red" : best_experiment["dim_red"],
                        "reduction_parameters" : best_experiment["reduction_params"],
                        "clustering" : best_experiment["clustering"],
                        "penalty" : best_experiment["penalty"],
                        "penalty_range" : best_experiment["penalty_range"],
                        # Important things
                        "classification_lvl": class_lvl,
                        "lvlm": model,
                        "prompt": prompt,
                        "eval_method": eval_method,
                        "best_score": best_experiment["score_w_penalty"] if "noise" in best_experiment["eval_method"] else best_experiment["score_w/o_penalty"], 
                        # Metrics
                        "homogeneity_global": quality_results["homogeneity_global"].iloc[0],
                        "entropy_global": quality_results["entropy_global"].iloc[0],
                        "quality_metric":quality_results["quality_metric"].iloc[0],
                        "homogeneity_global_w_noise": quality_results["homogeneity_global_w_noise"].iloc[0],
                        "entropy_global_w_noise": quality_results["entropy_global_w_noise"].iloc[0],
                        "quality_metric_w_noise":quality_results["quality_metric_w_noise"].iloc[0]
                    })


                    lvm_lvlm_metric.plot_cluster_categories_3()

    df_results = pd.DataFrame(result_list)
    df_results.to_csv("results.csv",sep=";")
