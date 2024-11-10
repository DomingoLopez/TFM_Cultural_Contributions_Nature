from pathlib import Path
import pickle
import shutil
import pandas as pd
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import requests
import time
import os
import sys

from src.utils.image_loader import ImageLoader
#from loguru import logger


class LlavaInference():
    """
    LlavaInference allows us to deploy selected Llava model (locally or in NGPU - UGR, but without automation yet)
    We start with Llava1.5-7b params. It can download model, and do some inference given some images and text prompt as inputs.
    """
    def __init__(self, 
                 images_dict_format: dict,
                 verbose= False):
        """
        Loads images from every cluster in order to do some inference on llava on ugr gpus
        Args:
            images_cluster_dict (dict)
        """

        # Setup logging
        # logger.remove()
        # if verbose:
        #     logger.add(sys.stdout, level="DEBUG")
        # else:
        #     logger.add(sys.stdout, level="INFO")


        # Base dir for moving images from every cluster. 
        # Just in case we decide to move this to ngpu wihout moving whole project.
        self.base_dir = Path(__file__).resolve().parent / "cluster_images/"
        self.results_dir = Path(__file__).resolve().parent / "results/"
        self.results_object = Path(__file__).resolve().parent / "results/result.pkl"
        self.classification_lvls_dir = Path(__file__).resolve().parent / "classification_lvls/"
        # Remove all images or files in that cluster images base dir
        shutil.rmtree(self.base_dir, ignore_errors=True)
        # Create dirs after cleaning
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        # Load categories (Try lvl 3)
        self.categories = pd.read_csv(os.path.join(self.classification_lvls_dir,"classification_level_3.csv"), header=None, sep=";").iloc[:, 0].tolist()


        # Initialize images_dict_format
        if images_dict_format is None:
            self.images_dict_format = self.load_images_from_base_dir()
        else:
            self.images_dict_format = images_dict_format


        # logger.info("Created LlavaControler. Cleaning cluster images folder")




    def load_images_from_base_dir(self):
        """
        Loads images from base directory, creating a dictionary with subfolder names as keys
        and lists of image paths as values.
        """
        images_dict = {}
        for subfolder in self.base_dir.iterdir():
            if subfolder.is_dir():
                image_loader = ImageLoader(subfolder)
                images_dict[subfolder.name] = [str(img_path) for img_path in image_loader.find_images()]
        return images_dict




    def createClusterDirs(self):
        """
        Create a dir for every cluster given in dictionary of images. 
        This is how we are gonna send that folder to ugr gpus
        """
        # logger.info("Copying images from Data path to cluster dirs")
        # For every key (cluster index)
        try:
            for k,v in self.images_dict_format.items():
                # Create folder if it doesnt exists
                cluster_dir = os.path.join(self.base_dir, str(k)) 
                os.makedirs(cluster_dir, exist_ok=True)
                # For every path image, copy that image from its path to cluster folder
                for path in v:
                    shutil.copy(path, cluster_dir)
        except (os.error) as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)



    def run(self):
        """
        Run Llava inference for every image in each subfolder of the base path.
        Store % success classifying every cluster in the same category.
        """
        # Initialize processor and model
        processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.to("cuda:0")

        # DataFrame to store results
        results = []

        # Iterate over each subfolder in base_dir
        for cluster_name, image_paths in self.images_dict_format.items():
            # ignore noise first
            if cluster_name != -1:
                #logger.info(f"Processing cluster: {cluster_name}")
                
                start_time = time.time()
                category_counts = {}

                for image_path in image_paths:
                    image = Image.open(image_path)
                    
                    # Construct prompt
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Classify this image in one of these categories: " + ", ".join(self.categories)},
                                {"type": "image", "image": image},  
                            ],
                        },
                    ]
                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
                    
                    # Generate inference
                    output = model.generate(**inputs, max_new_tokens=100)
                    classification_result = processor.decode(output[0], skip_special_tokens=True)

                    # Count classification results for predominant category analysis
                    if classification_result in category_counts:
                        category_counts[classification_result] += 1
                    else:
                        category_counts[classification_result] = 1

                # Calculate success percentage and predominant category
                total_images = len(image_paths)
                predominant_category = max(category_counts, key=category_counts.get)
                predominant_count = category_counts[predominant_category]
                success_percent = (predominant_count / total_images) * 100
                inference_time_per_cluster = time.time() - start_time

                # Append to results
                results.append({
                    "cluster": cluster_name,
                    "total_img": total_images,
                    "success_percent": success_percent,
                    "predominant_category": predominant_category,
                    "inference_time_per_cluster": inference_time_per_cluster
                })

        # Create DataFrame with results
        results_df = pd.DataFrame(results)
        
        # Save to file or display
        results_path = self.results_dir / "inference_results.csv"
        results_df.to_csv(results_path, index=False)
        pickle.dump(results_df, open(self.results_object, "wb"))
        #logger.info(f"Results saved to {results_path}")



    def test_llava():
        """
        Just some test in order to make sure Llava is working, locally and on ugr gpu 
        """
        # Load processor and model
        processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        # Move model to gpu. If not it is almost impossible unless we use LORA or some quantization, etc.
        model.to("cuda:0")

        # Example image, just to make user it works
        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        image = Image.open(requests.get(url, stream=True).raw)

        # Prompt
        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this image in one of these categories: Nature, Urban, Rural or Others"},
                {"type": "image", "image": image},  
                ],
            },
        ]

        # Apply chat template 
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # process imputs making sure cuda is available
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        # Do inference (measure inference time too)
        start_time = time.time()
        output = model.generate(**inputs, max_new_tokens=100)
        end_time = time.time()


        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} s")

        # Decoding result
        print(processor.decode(output[0], skip_special_tokens=True))




if __name__ == "__main__":

    llava = LlavaInference(images_dict_format=None)
    llava.run()
