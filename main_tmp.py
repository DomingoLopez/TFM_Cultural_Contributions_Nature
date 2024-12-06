from collections import Counter
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from loguru import logger

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
    
    file_path = "src/llava_inference/results"
    experiment_files = Path(file_path).rglob("*.csv")

    for file in experiment_files:
        # Leer el archivo CSV
        df = pd.read_csv(file, sep=';')

        # Eliminar la columna 'cluster'
        df.drop(columns=['cluster'], inplace=True)

        # Transformar la columna 'img' para que solo contenga el nombre de la imagen
        df['img'] = df['img'].apply(lambda x: x.split('/')[-1])

        # Guardar el archivo procesado, sobreescribiendo el original
        df.to_csv(file, index=False, sep=';')

        print(f"Procesado: {file}")
