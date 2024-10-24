import pandas as pd

from src.utils.image_loader import ImageLoader
from src.dinov2_inference.dinov2_inference import Dinov2Inference



if __name__ == "__main__":
    # Finding images
    image_loader = ImageLoader(folder="./data/Small_Data")
    images = image_loader.find_images()
    # Loading images and getting embeddings
    dinomodel = Dinov2Inference(model_name="small", images=images)
    embeddings = dinomodel.run()
    # TODO: Load embeddings into df. 
    
    
    
    
    
    
    # TODO: Dimensionality reduction (or not)
    # TODO: Apply Clustering techniques
    