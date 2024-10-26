import pandas as pd

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
    embeddings_after_dimred = eda.run_eda(dimensions=2, dim_reduction = "cvae", show_plots=False)
    # TODO: Apply Clustering techniques
    