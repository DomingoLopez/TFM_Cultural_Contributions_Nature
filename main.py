from src.image_loader import ImageLoader
from src.dinov2_inference import Dinov2Inference


if __name__ == "__main__":
    # Finding images
    image_loader = ImageLoader(folder="./data")
    images = image_loader.find_images()
    # Loading images and getting embeddings
    embeddings = Dinov2Inference(model="small", images=images)
    