import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json

from src.clustering.clust_hdbscan import HDBSCANClustering
from src.clustering.clustering_factory import ClusteringFactory
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
    # Finding images
    # image_loader = ImageLoader(folder="./data/Small_Data")
    image_loader = ImageLoader(folder="./data/Data")
    images = image_loader.find_images()
    # Loading images and getting embeddings
    dinomodel = Dinov2Inference(model_name="small", images=images, disable_cache=False)
    embeddings = dinomodel.run()

    # Load json file with all experiments
    with open('src/experiment/json/all_experiments.json', 'r') as f:
        experiments_config = json.load(f)

    for config in experiments_config:
        optimizer = config.get("optimizer", "optuna")
        dim_red_range = config.get("dim_red_range", [2, 15])
        scalers = config.get("scalers", ["standard", "minmax", "robust", "maxabs"])
        dim_red = config.get("dim_red", "umap")
        clustering = config.get("clustering", "hdbscan")
        eval_method = config.get("eval_method", "silhouette")
        penalty = config.get("penalty", None)
        penalty_range = config.get("penalty_range", None)
        cache = config.get("cache", False)
        # Make and Run Experiment
        experiment = Experiment(
            embeddings,
            optimizer,
            dim_red,
            dim_red_range,
            scalers,
            clustering,
            eval_method,
            None if penalty == "" else penalty,
            None if penalty_range== "" else penalty_range,
            False if cache == 0 else True
        )
        experiment.run_experiment()


    # ##############################################################
    # BIG STUDY
    # ##############################################################

    # # Variables initialization
    # scalers = ["standard","minmax","robust","maxabs"]
    # dim_red = "umap"
    # clustering = "agglomerative"
    # eval_method = "silhouette"
    # penalty = None 
    # penalty_range = None
    # cache = True
    # result_dir_cache_path = Path(__file__).resolve().parent / f"cache/results_optuna/{clustering}/{eval_method}_penalty_{penalty}_images_{len(images)}"
    # os.makedirs(result_dir_cache_path, exist_ok=True)
    # result_file_cache_path = Path(__file__).resolve().parent / result_dir_cache_path / "result.pkl"
    # result_file_cache_path_csv = Path(__file__).resolve().parent / result_dir_cache_path / "result.csv"
    # results = []
    
    # # If file with results doesnt exists
    # if not os.path.isfile(result_file_cache_path) or not cache:
    #     for scaler in scalers:
    #         embeddings_scaled = eda.run_scaler(scaler)
    #         for dim in range(15, 25):
    #             embeddings_after_dimred = eda.run_dim_red(embeddings_scaled, dimensions=dim, dim_reduction=dim_red, show_plots=False)
    #             clustering_model = ClusteringFactory.create_clustering_model(clustering, embeddings_after_dimred)
    #             # Execute optuna
    #             study = clustering_model.run_optuna(evaluation_method=eval_method, n_trials=100, penalty=penalty, penalty_range=penalty_range)
    #             # Access best trial n_cluster
    #             best_trial = study.best_trial
    #             n_clusters_best = best_trial.user_attrs.get("n_clusters", None)  # Extract clusters
    #             centers_best = best_trial.user_attrs.get("centers", None)  # Extract centers
    #             score_best = best_trial.user_attrs.get("score_original", None)  # Extract original score
    #             # Store results
    #             results.append({
    #                 "scaler": scaler,
    #                 "dim_reduction":dim_red,
    #                 "dimension": dim,
    #                 "n_clusters": n_clusters_best,
    #                 "best_params": str(study.best_params),
    #                 "centers": centers_best,
    #                 "best_value": study.best_value,
    #                 "best_original_value": score_best
    #             })

    #     # Store results as dataframe and csv in cache
    #     results_df = pd.DataFrame(results)
    #     results_df.to_csv(result_file_cache_path_csv,sep=";")
    #     # Save study as object in cache.
    #     results_cache_path = ""
    #     pickle.dump(
    #         results_df,
    #         open(str(result_file_cache_path), "wb"),
    #     )
    # else:
    #     try:
    #         results_df = pickle.load(
    #             open(str(result_file_cache_path), "rb")
    #         )
    #         results_df.to_csv(result_file_cache_path_csv,sep=";")
    #     except:
    #         FileNotFoundError("Couldnt find provided file with results from experiments. Please, ensure that file exists.")

    # With results we can make final experiment with desired params
    # For example give me experiment where it got 4 clusters with best
    # value of its metric (davies or silhouette)
    # For now, we could take best result











    # if eval_method == "silhouette":
    #     best_experiment = results_df.loc[results_df['best_value_w/o_penalty'].idxmax()]
    # else:
    #     best_experiment = results_df.loc[results_df['best_value_w/o_penalty'].idxmin()]
        
    # # Gest data from best experiment
    # best_centers = best_experiment["centers"]
    # best_scaler = best_experiment["scaler"]
    # best_dimension = best_experiment["dimension"]
    # best_dim_red = best_experiment["dim_reduction"]
    # # Convert to dictionary
    # best_params = best_experiment["best_params"].replace("'", "\"").replace("True", "true").replace("False", "false")

    # print(f"Best parameters raw: {best_params}")

    # try:
    #     best_params_dict = json.loads(best_params)
    # except json.JSONDecodeError as e:
    #     print("Error decoding JSON params:", e)
    #     print("Parameters are not correct:", best_params)
    #     raise


    # # Use single experiment 
    # # TODO: ESTO se puede poner como un print o el __str__ del experiment una vez completado
    # # HACER UN MÉTODO SHOW_RESULTS que muestre de forma formateada y con plots los resultados. 
    # # single_experiment = clustering_model.run_single_experiment()
    # eda = EDA(embeddings=embeddings, verbose=False)
    # embeddings_scaled = eda.run_scaler(best_scaler)
    # embeddings_after_dimred = eda.run_dim_red(embeddings_scaled, dimensions=best_dimension, dim_reduction=best_dim_red, show_plots=False)
    # clustering_model = ClusteringFactory.create_clustering_model(clustering, embeddings_after_dimred)
    # # Run single Experiment
    # labels, centers, score = clustering_model.run_single_experiment(best_params_dict,eval_method)
    
    # unique_labels, counts = np.unique(labels, return_counts=True)
    # conteo_clusters = dict(zip(unique_labels, counts))
    # print(f"Mejores parámetros: {best_params_dict}")
    # print(f"Score {eval_method} tras single experiment: {score}")
    # print(f"N. Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
    # print(f"Conteo de imágenes por cluster: {conteo_clusters}")
    # # Plot experiment  
    # # TODO: move this pca calculation to eda object. Give more sense to eda object
    # pca_df, pca_centers = clustering_model.do_PCA_for_representation(embeddings_after_dimred, centers)
    # clustering_model.plot_single_experiment(pca_df, labels, pca_centers, i=0, j=1)
    # # Obtain knn image index for each cluster
    # # Lets suppose that the dim reduction is the same for every case, and the centers are the same.
    # # Lets calculate similarities
    # knn_similarity_df = clustering_model.find_clustering_knn_points(5, best_params_dict.get("metric"), best_centers, labels)
    # cosine_similarity_df = clustering_model.find_clustering_cosine_similarity_points(5, best_centers, labels)
    # # print closests points to center based on knn
    # print("Closest points to center based on knn:")
    # print(knn_similarity_df)
    # print("\nClosest points to center based on cosine similarity:")
    # print(cosine_similarity_df)

    # print("\n\n[KNN] - Showing images related (x nn) to each cluster:")
    # show_images_per_cluster(images, knn_similarity_df)
    # print("\n\n[COSINE] - Showing images related (x nn) to each cluster:")
    # show_images_per_cluster(images, cosine_similarity_df)
    
    
