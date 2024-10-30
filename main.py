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

    # TODO: NORMALIZAR EN EL EDA
    # Create Eda object and apply or not dim reduction
    eda = EDA(embeddings=embeddings, verbose=False)
    
    results = []
    for i in range(3,20):
        embeddings_after_dimred = eda.run_dim_red(dimensions=i, dim_reduction = 'umap', show_plots=False)
        clustering_model = ClusteringFactory.create_clustering_model("agglomerative", embeddings_after_dimred)
        study = clustering_model.run_optuna(evaluation_method="silhouette", n_trials=500)
        print(study)
        results.append((i, study.best_params, study.best_value))
        
        
    # # Create clustering factory and kmeans
    # # TODO: Here we could pass a eda object to Clustering creation, so it would know how many dimensiones
    # # do we have and put that in another subfolder with results, or even add that to path name of results.
    # clustering_model = ClusteringFactory.create_clustering_model("agglomerative", embeddings_after_dimred)
    # # Run Clustering
    # study = clustering_model.run_optuna(evaluation_method="silhouette", n_trials=500)
    # # print("Clustering complete. Results available in results/modelname/timestamp")
    
    print(results)
    