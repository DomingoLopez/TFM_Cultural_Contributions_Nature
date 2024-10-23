import http
from pathlib import Path
import pickle
import sys
import json

import appdirs
from loguru import logger
import numpy as np
import PIL
from PIL import Image, ImageOps
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
    def __init__(self, model_name="small", 
                 model_path=None, 
                 images=None, 
                 disable_cache = True, 
                 database = "./results/embeddings/", 
                 verbose=False):
        
        # Attr initialization
        with open("json/dinov2_sizes.json",'r') as model_sizes:
            self.model_name = json.load(model_sizes).get(model_name)
        self.model_folder = "facebookresearch/dinov2" if model_path is None else model_path
        self.model_source = "github" if model_path is None else "local"
        # Folder where cache exists
        self.database = database
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
            cache_root_folder = Path(
                appdirs.user_cache_dir(appname="dinov2_inference", appauthor="domi")
            )
            cache_root_folder.mkdir(parents=True, exist_ok=True)
            self.embeddings_cache_path = cache_root_folder / (
                Path(database).name + "_" + model_name + ".pkl"
            )
            logger.debug(f"{cache_root_folder=}, {self.embeddings_cache_path=}")


    def __extract_single_image_feature(self, image):
        """
        Extract backbone feature of dino v2 model on a single image
        """
        net_input = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            feature = self.model(net_input).squeeze().numpy()
        return feature
    
    def __glob_images(self, path):
        """
        Find all image files in path (for cache only)
        """
        return (
            list(path.rglob("*.jpg"))
            + list(path.rglob("*.JPG"))
            + list(path.rglob("*.jpeg"))
            + list(path.rglob("*.png"))
            + list(path.rglob("*.bmp"))
        )
    
    
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
        """
        
        # Extract images from cache
        database_img_paths = self.__glob_images(Path(self.database))

        if len(database_img_paths) < 1:
            logger.warning("database does not contain images, exit")
            return
        print("Path----",database_img_paths)

        # Extract embeddings for images or load from cache
        if self.disable_cache or not self.embeddings_cache_path.exists():
            logger.info("Preparing embeddings")
            embeddings = self.__generate_embeddings()
            if not self.disable_cache:
                pickle.dump(
                    embeddings,
                    open(str(self.embeddings_cache_path), "wb"),
                )
        else:
            logger.info(
                f"Load cached database features from {self.embeddings_cache_path}"
            )
            embeddings = pickle.load(
                open(str(self.embeddings_cache_path), "rb")
            )

        return embeddings


    def calculate_distance(self, query_feature, database_features):
        cosine_distances = [
            np.dot(query_feature, feature)
            / (np.linalg.norm(query_feature) * np.linalg.norm(feature))
            for feature in database_features
        ]
        return cosine_distances

    def calculate_torch_distance(self, query_feature, database_features):
        # #distance = []
        # #for feature in database_features:
        #     #distance.append(
        # x1 = query_feature
        # x2 = database_features[0]
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6) 
        # print("x1", len(x1), "x2",len(x2))
        # cos = cos([x1],[x2]) 
        # dist = 1 - cos


        # print("torch..", dist)
        
        # return distance
        pass

    def save_result(
        self,
        args,
        query_image,
        query_path,
        database_img_paths,
        closest_indices,
        sorted_distances,
    ):
        img_save_folder = (
            Path(args.output_root)
            / Path(args.database).name
            / Path(self.model_name).name
        )
        img_save_path = img_save_folder / (
            query_path.stem + "_output" + query_path.suffix
        )
        logger.info(f"Save results to {img_save_path}")

        img_save_folder.mkdir(parents=True, exist_ok=True)

        # pad and resize image, in order to combine query and retrieved image in a single image
        query_image = self.process_image_for_visualization(args, query_image)

        vis_img_list = [query_image]
        for idx, img_idx in enumerate(closest_indices):
            img_path = database_img_paths[img_idx]
            similarity = sorted_distances[idx]
            logger.debug(
                f"{idx}th similar image is {img_path}, similarity is {similarity}"
            )
            cur_img = Image.open(img_path)
            cur_img = self.process_image_for_visualization(args, cur_img)
            vis_img_list.append(cur_img)

        x_offset = 0
        out_img = Image.new(
            "RGB", (args.size * (self.top_k + 1) + args.margin * self.top_k, args.size)
        )
        for img in vis_img_list:
            out_img.paste(img, (x_offset, 0))
            x_offset += img.width + args.margin

        out_img.save(str(img_save_path))

    def process_image_for_visualization(self, args, img):
        # #"""pad then resize image to target size"""
        # #width, height = img.size
        # #if width > height:
        # #    new_width = args.size
        # #    new_height = int((new_width / width) * height)
        # #else:
        # #    new_height = args.size
        # #    new_width = int((new_height / height) * width)

        # #img = img.resize((new_width, new_height))

        # #width, height = img.size
        # #target_size = args.size
        # #width, height = img.size
        # #delta_w = target_size - width
        # #delta_h = target_size - height
        # #padding = (
        # #    delta_w // 2,
        # #    delta_h // 2,
        # #    delta_w - (delta_w // 2),
        # #    delta_h - (delta_h // 2),
        # #)

        # ## fill with gray color
        # #padded_img = ImageOps.expand(img, padding, fill=0)
        # return padded_img
        pass

    def extract_single_image_feature(self, image):
        """extract backbone feature of dino v2 model on a single image"""
        net_input = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            feature = self.model(net_input).squeeze().numpy()

        print("FEATURE....\n", feature)
        return feature
