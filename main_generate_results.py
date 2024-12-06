import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from loguru import logger

from src.experiment.experiment import Experiment
from src.multimodal_clustering_metric.multimodal_clustering_metric import MultiModalClusteringMetric


import matplotlib.pyplot as plt


def get_eval_method_value(exp_id):
    base_path = "src/experiment/clusters"

    # Construimos la ruta base
    experiment_path = os.path.join(base_path, f"experiment_{exp_id}")
    # Verificamos que la ruta existe
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"La ruta {experiment_path} no existe.")
    # Buscamos subcarpetas
    subfolders = [f for f in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, f))]
    # Tomamos la primera subcarpeta (puedes adaptar si hay más de una)
    if not subfolders:
        raise ValueError(f"No se encontraron subcarpetas en {experiment_path}.")
    target_folder = subfolders[0]  # Tomamos la primera
    print(f"Subcarpeta encontrada: {target_folder}")
    # Dividimos la subcarpeta por "_" y tomamos la última parte
    try:
        index_value = int(target_folder.split("_")[1])
        score_value = float(target_folder.split("_")[-1])  # Convertimos a float
    except ValueError:
        raise ValueError(f"No se pudo extraer un valor numérico de la subcarpeta: {target_folder}")
    
    return index_value, score_value


def get_quality_metrics(class_lvl, prompt, id, use_noise_in_metric):
    base_path = "src/experiment/clusters"

    # Construimos la ruta base
    experiment_path = os.path.join(base_path, f"experiment_{exp_id}")
    # Verificamos que la ruta existe
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"La ruta {experiment_path} no existe.")
    # Buscamos subcarpetas
    subfolders = [f for f in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, f))]
    # Tomamos la primera subcarpeta (puedes adaptar si hay más de una)
    if not subfolders:
        raise ValueError(f"No se encontraron subcarpetas en {experiment_path}.")
    target_folder = subfolders[0]  # Tomamos la primera
    print(f"Subcarpeta encontrada: {target_folder}")
    # Dividimos la subcarpeta por "_" y tomamos la última parte
    try:
        index_value = int(target_folder.split("_")[1])
        score_value = float(target_folder.split("_")[-1])  # Convertimos a float
    except ValueError:
        raise ValueError(f"No se pudo extraer un valor numérico de la subcarpeta: {target_folder}")
    
    return index_value, score_value




def create_results_csv():
    file = "src/experiment/json/experiments_optuna_silhouette_umap.json"

    with open(file, 'r') as f:
        experiments_config = json.load(f)

    classification_lvl = [3]
    prompts = [1,2]
    models = ["llava1-5_7b","llava1-6_7b","llava1-6_13b"]
    for class_lvl in classification_lvl:
        for model in models:
            for prompt in prompts:
                result_list = []
                for config in experiments_config:
                    id = config.get("id")
                    dino_model = config.get("dino_model","small")
                    optimizer = config.get("optimizer", "optuna")
                    optuna_trials = config.get("optuna_trials", None)
                    normalization = config.get("normalization", True)
                    scaler = config.get("scaler", None)
                    clustering = config.get("clustering", "hdbscan")
                    eval_method = config.get("eval_method", "silhouette")
                    penalty = config.get("penalty", None)
                    penalty_range = config.get("penalty_range", None)
                    dim_red = config.get("dim_red", None)

                    # Get experiment index value
                    # Hay que ir al csv, con los filtros y traerse reduction params tb, etc
                    index, score, reduction_parameters = get_exp_results(id)

                    

                    for use_noise_in_metric in [True, False]:
                        homogeneity_global, entropy_global, quality_metric = get_quality_metrics(class_lvl, prompt, id, use_noise_in_metric)

                        result_list.append({
                            "experiment_id" : id,
                            "best_experiment_index": index,
                            "dino_model" : dino_model,
                            "normalization" : normalization,
                            "scaler" : scaler,
                            "dim_red" : dim_red,
                            "reduction_parameters" : reduction_parameters,
                            "clustering" : clustering,
                            "penalty" : penalty,
                            "penalty_range" : penalty_range,
                            # Important things
                            "classification_lvl": class_lvl,
                            "lvlm": model,
                            "prompt": prompt,
                            "eval_method": eval_method,
                            "best_score": score, 
                            # Metrics
                            "use_noise" : use_noise_in_metric,
                            "homogeneity_global": homogeneity_global,
                            "entropy_global": entropy_global,
                            "quality_metric":quality_metric
                        })



if __name__ == "__main__": 
    
    create_results_csv()

    


