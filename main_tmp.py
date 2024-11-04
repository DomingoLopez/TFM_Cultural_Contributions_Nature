from collections import Counter
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from loguru import logger

from src.clustering.clust_hdbscan import HDBSCANClustering
from src.clustering.clustering_factory import ClusteringFactory
from src.clustering_plot.clust_plot import ClusteringPlot
from src.experiment.experiment import Experiment
from src.utils.image_loader import ImageLoader
from src.dinov2_inference.dinov2_inference import Dinov2Inference
from src.eda.eda import EDA

import matplotlib.pyplot as plt
import cv2


def show_images_per_cluster(images, knn_cluster_result_df):
    n_clusters = knn_cluster_result_df.shape[1]  # Número de clusters
    n_images_per_cluster = knn_cluster_result_df.shape[0]  # Número de imágenes por cluster
    
    # Crear una figura con subplots para cada cluster y sus imágenes correspondientes
    fig, axs = plt.subplots(n_clusters, n_images_per_cluster, figsize=(15, n_clusters * 3), squeeze=False)
    fig.suptitle("Closest Images to Cluster Centers", fontsize=16)
    
    for cluster_idx in range(n_clusters):
        cluster_name = f"Cluster_{cluster_idx}"
        for img_idx in range(n_images_per_cluster):
            image_index = knn_cluster_result_df[cluster_name].iloc[img_idx]  # Índice de la imagen en el array 'images'
            img_path = images[image_index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para matplotlib
            
            # Asegurarse de que `axs` siempre sea bidimensional
            ax = axs[cluster_idx, img_idx]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{cluster_name} - Img {img_idx + 1}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar el layout para el título
    plt.show()




if __name__ == "__main__":
    
    try:
        # Cargar el archivo usando un contexto para garantizar su cierre
        with open("src/experiment/results/kmeans/optuna/dim_red_umap/silhouette_penalty_None.pkl", "rb") as f:
            result = pickle.load(f)
        
        # Asegurarnos de que `result` es un DataFrame y contiene la columna "labels"
        if isinstance(result, pd.DataFrame) and "labels" in result.columns:
            # Aplicar Counter a cada fila en la columna "labels"
            label_counter_series = result["labels"].apply(lambda x: Counter(x) if isinstance(x, (list, np.ndarray)) else None)
            
            # Obtener el índice de la columna "labels"
            labels_index = result.columns.get_loc("labels") + 1
            
            # Insertar la nueva columna "label_counter" justo después de "labels"
            result.insert(labels_index, "label_counter", label_counter_series)
        else:
            print("Error: 'result' no es un DataFrame o no contiene la columna 'labels'.")

    except FileNotFoundError:
        print("Error: El archivo especificado no existe.")
    except pickle.UnpicklingError:
        print("Error: No se pudo cargar el archivo. Puede estar dañado o no es un archivo de pickle válido.")
    except KeyError:
        print("Error: La clave 'labels' no se encontró en el diccionario 'result'.")

    pickle.dump(
        result,
        open(str("src/experiment/results/kmeans/optuna/dim_red_umap/silhouette_penalty_None.pkl"), "wb")
    )

    result.to_csv("src/experiment/results/kmeans/optuna/dim_red_umap/silhouette_penalty_None.csv", sep=";")
