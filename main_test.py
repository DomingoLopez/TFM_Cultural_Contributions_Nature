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
from src.utils.image_loader import ImageLoader
from src.dinov2_inference.dinov2_inference import Dinov2Inference
from src.clustering_plot.clust_plot import ClusteringPlot
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
    # Finding images
    # image_loader = ImageLoader(folder="./data/Small_Data")
    image_loader = ImageLoader(folder="./data/Small_Data")
    images = image_loader.find_images()
    # Loading images and getting embeddings
    dinomodel = Dinov2Inference(model_name="small", images=images, disable_cache=False)
    embeddings = dinomodel.run()

    optimizer = "optuna"
    dim_red_range = [2, 3]
    scalers =  ["standard"]
    dim_red =  "umap"
    clustering = "kmeans"
    eval_method =  "silhouette"
    penalty =  None
    penalty_range =  None
    cache =  True

    experiment = Experiment(
        embeddings,
        optimizer,
        dim_red,
        dim_red_range,
        scalers,
        clustering,
        eval_method,
        penalty,
        penalty_range,
        cache
    )
    
    experiment.run_experiment()
    
    plot = ClusteringPlot(experiment=experiment)
    plot.show_best_silhouette(show_plots=True)
    plot.show_best_scatter_with_centers(show_plots=True)


    

    
    
