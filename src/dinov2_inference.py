import http
from pathlib import Path
import pickle
import sys
import json

from loguru import logger
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class Dinov2Inference:
    """
    Inference from DinoV2 from a set of images.
    Includes preprocessing, noise clean up, transformations, etc for
    the correct load of images to Dinov2.
    """
    def __init__(self, 
                 model_name="small", 
                 model_path=None, 
                 images=None, 
                 disable_cache = False, 
                 verbose=False):
        
        # Initializing model
        json_sizes_path = Path(__file__).resolve().parent / "json/dinov2_sizes.json"
        with open(json_sizes_path,'r') as model_sizes:
            self.model_name = json.load(model_sizes).get(model_name)

        self.model_folder = "facebookresearch/dinov2" if model_path is None else model_path
        self.model_source = "github" if model_path is None else "local"
        self.disable_cache = disable_cache
        
        # Validate image list
        if not isinstance(images, list):
            raise TypeError(f"Expected 'images' to be a list, but got {type(images).__name__} instead.")
        if len(images) < 1:
            raise ValueError("The 'images' list must contain at least one image.")
        self.images = images  

        # Setup logging
        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        try:
            logger.info(f"loading {self.model_name=} from {self.model_folder=}")
            self.model = torch.hub.load(
                self.model_folder,
                self.model_name,
                source=self.model_source,
            )
        except FileNotFoundError:
            logger.error(f"load model failed. please check if {self.model_folder=} exists")
            sys.exit(1)
        except http.client.RemoteDisconnected:
            logger.error(
                "connect to github is reset. maybe set --model-path to $HOME/.cache/torch/hub/facebookresearch_dinov2_main ?"
            )
            sys.exit(1)

        # Setup model in eval mode.
        self.model.eval()

        # Construct image tranforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Generate cache folder if cache up
        if not self.disable_cache:
            # cache_root_folder = Path(
            #     appdirs.user_cache_dir(appname="dinov2_inference", appauthor="domi")
            # )
            project_root = Path(__file__).resolve().parent.parent  # Un nivel hacia arriba desde src/
            cache_root_folder = project_root / "cache"
            cache_root_folder.mkdir(parents=True, exist_ok=True)
            self.embeddings_cache_path = cache_root_folder / (
                "embeddings_" + model_name + ".pkl"
            )
            logger.debug(f"{cache_root_folder=}, {self.embeddings_cache_path=}")



    def __extract_single_image_feature(self, image):
        """
        Extract backbone feature of dino v2 model on a single image
        and return a numpy array from that, ready to apply other algorithms
        """
        net_input = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            feature = self.model(net_input).squeeze().numpy()
        return feature
    
    
    
    def __generate_embeddings(self):
        """
        Generate all embeddings for loaded images.
        """
        embeddings = []
        for img_path in tqdm(self.images):
            print("image :", str(img_path))
            img = Image.open(str(img_path)).convert("RGB")
            feature = self.__extract_single_image_feature(img)
            embeddings.append(feature)
        return embeddings
        


    def run(self):
        """
        Execute inference for loaded images 
        (using or not cache to avoid inference, for testing purposes)
        """
        # If we dont want cache
        if self.disable_cache or not self.embeddings_cache_path.exists():
            logger.info("Preparing embeddings")
            embeddings = self.__generate_embeddings()
            if not self.disable_cache:
                pickle.dump(
                    embeddings,
                    open(str(self.embeddings_cache_path), "wb"),
                )
        # If we want cache
        else:
            logger.info(
                f"Load cached database features from {self.embeddings_cache_path}"
            )
            embeddings = pickle.load(
                open(str(self.embeddings_cache_path), "rb")
            )
        return embeddings
