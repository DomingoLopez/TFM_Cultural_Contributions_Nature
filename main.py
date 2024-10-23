from src.image_loader import ImageLoader
from src.dinov2_inference import Dinov2Inference


if __name__ == "__main__":
    # Finding images
    image_loader = ImageLoader(folder="./data/test/")
    images = image_loader.find_images()
    # Loading images and getting embeddings
    dinomodel = Dinov2Inference(model_name="small", images=images)
    embeddings = dinomodel.run()
    print(len(embeddings))
    # TODO: Load embeddings into df. 
    # TODO: Dimensionality reduction (or not)
    # TODO: Apply Clustering techniques
    