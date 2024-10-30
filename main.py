import pandas as pd

from src.clustering.clustering_factory import ClusteringFactory
from src.utils.image_loader import ImageLoader
from src.dinov2_inference.dinov2_inference import Dinov2Inference
from src.eda.eda import EDA



if __name__ == "__main__":
    # Finding images
    image_loader = ImageLoader(folder="./data/Small_Data")
    images = image_loader.find_images()
    # Loading images and getting embeddings
    dinomodel = Dinov2Inference(model_name="small", images=images)
    embeddings = dinomodel.run()

    # t = pd.DataFrame(embeddings)
    # print(t .head())
    # print(t.describe())
    # print(t.info())

    # Create Eda object and apply or not dim reduction
    eda = EDA(embeddings=embeddings, verbose=False)
    #embeddings_scaled = eda.run_scaler()
    
    scalers = ["standard","minmax","robust","maxabs"]
    results = []
    for scaler in scalers:
        embeddings_scaled = eda.run_scaler(scaler)
        for dim in range(3, 15):
            embeddings_after_dimred = eda.run_dim_red(embeddings_scaled, dimensions=dim, dim_reduction='umap', show_plots=False)
            clustering_model = ClusteringFactory.create_clustering_model("hdbscan", embeddings_after_dimred)
            
            # Ejecuta Optuna y almacena el estudio
            study = clustering_model.run_optuna(evaluation_method="silhouette", n_trials=100)
            
            # Accede al número de clústeres en el mejor ensayo
            best_trial = study.best_trial
            n_clusters_best = best_trial.user_attrs.get("n_clusters", None)  # Extrae el número de clústeres

            # Almacena resultados
            results.append({
                "scaler": scaler,
                "dimension": dim,
                "n_clusters": n_clusters_best,
                "best_params": str(study.best_params),
                "best_value": study.best_value
            })

    # Convierte resultados en DataFrame y guarda en CSV
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv("resultado.csv", sep=";")
        
    # # Create clustering factory and kmeans
    # # TODO: Here we could pass a eda object to Clustering creation, so it would know how many dimensiones
    # # do we have and put that in another subfolder with results, or even add that to path name of results.
    # clustering_model = ClusteringFactory.create_clustering_model("agglomerative", embeddings_after_dimred)
    # # Run Clustering
    # study = clustering_model.run_optuna(evaluation_method="silhouette", n_trials=500)
    # # print("Clustering complete. Results available in results/modelname/timestamp")
