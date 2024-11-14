import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from loguru import logger


import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import normalize




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


def rename_umap_files(folder_path):
    # Recorre todos los archivos de la carpeta
    for filename in os.listdir(folder_path):
        # Verifica si el archivo tiene el formato "umap_n_components..."
        if filename.startswith("umap_n_components"):
            # Construye el nuevo nombre de archivo
            new_filename = filename.replace("umap_", "umap_metric=euclidean_", 1)
            # Obtiene las rutas completas de los archivos
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_filename)
            # Renombra el archivo
            os.rename(old_filepath, new_filepath)
            print(f'Renombrado: "{filename}" a "{new_filename}"')



if __name__ == "__main__":
    
    # Cargar los embeddings desde el archivo
    embeddings = pickle.load(open("src/dinov2_inference/cache/embeddings_dinov2_vits14_5066.pkl", "rb"))

    # Calcular la norma L2 de cada embedding
    l2_norms = np.linalg.norm(embeddings, axis=1)

    # Verificar si están aproximadamente normalizados a 1
    are_normalized = np.allclose(l2_norms, 1, atol=1e-6)
    print("Embeddings normalizados L2 (antes de normalizar):", are_normalized)

    # Normalizar manualmente para asegurar L2
    embeddings_normalized = normalize(embeddings, norm='l2')

    # Calcular la norma L2 de los embeddings normalizados
    l2_norms_normalized = np.linalg.norm(embeddings_normalized, axis=1)
    are_normalized_after = np.allclose(l2_norms_normalized, 1, atol=1e-6)
    print("Embeddings normalizados L2 (después de normalizar):", are_normalized_after)
    print(embeddings[:1])
    print(embeddings_normalized[:1])
    
    
