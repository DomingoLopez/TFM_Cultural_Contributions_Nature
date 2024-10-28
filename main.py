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
    # Create Eda object and apply or not dim reduction
    eda = EDA(embeddings=embeddings, verbose=False)
    embeddings_after_dimred = eda.run_eda(dimensions=9, dim_reduction = None, show_plots=False)
    # Create clustering factory and kmeans
    # TODO: Here we could pass a eda object to Clustering creation, so it would know how many dimensiones
    # do we have and put that in another subfolder with results, or even add that to path name of results.
    clustering_model = ClusteringFactory.create_clustering_model("kmeans", embeddings_after_dimred)
    # Run Clustering
    clustering_model.run()
    print("Clustering complete. Results available in results/modelname/timestamp")