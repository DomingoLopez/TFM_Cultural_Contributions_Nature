from pathlib import Path
import pickle
import shutil
import pandas as pd
#from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import requests
import time
import os
import sys
import matplotlib.pyplot as plt

#from src.utils.image_loader import ImageLoader
#from loguru import logger


class LlavaInference():
    """
    LlavaInference allows us to deploy selected Llava model (locally or in NGPU - UGR, but without automation yet)
    We start with Llava1.5-7b params. It can download model, and do some inference given some images and text prompt as inputs.
    """
    def __init__(self, 
                 images_dict_format: dict,
                 classification_lvl: str,
                 experiment_name: str,
                 cache: bool = True, 
                 verbose: bool = False):
        """
        Loads images from every cluster in order to do some inference on llava on ugr gpus
        Args:
            images_cluster_dict (dict)
            classification_lvl (str): Classification level to be used
            experiment_name (str): Name of the experiment for organizing results
        """
        
        # Base dir for moving images from every cluster.
        self.base_dir = Path(__file__).resolve().parent / "cluster_images/"
        self.results_dir = Path(__file__).resolve().parent / f"results/classification_lvl_{classification_lvl}/{experiment_name}"
        self.results_object = self.results_dir / "result.pkl"
        self.classification_lvls_dir = Path(__file__).resolve().parent / "classification_lvls/"
        
        # Ensure directories exist
        if images_dict_format is not None:
            shutil.rmtree(self.base_dir, ignore_errors=True)
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Load categories based on classification level
        self.classification_lvl = classification_lvl
        self.categories = pd.read_csv(os.path.join(self.classification_lvls_dir, f"classification_level_{self.classification_lvl}.csv"), header=None, sep=";").iloc[:, 0].tolist()
        self.cache = cache

        # Initialize images_dict_format
        self.images_dict_format = images_dict_format if images_dict_format is not None else self.load_images_from_base_dir()

        # Results
        self.result_df = None
        self.result_stats_df = None

        # logger.info("Created LlavaControler. Cleaning cluster images folder")




    def load_images_from_base_dir(self):
        """
        Loads images from base directory, creating a dictionary with subfolder names as keys
        and lists of image paths as values.
        """
        images_dict = {}
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        for subfolder in self.base_dir.iterdir():
            if subfolder.is_dir():
                # Find all image files in subfolder recursively and filter by extension
                image_paths = [img_path for img_path in subfolder.rglob('*') if img_path.suffix.lower() in image_extensions]
                
                # Convert paths to lowercase to avoid duplicates (especially relevant for Windows)
                unique_image_paths = {img_path.resolve().as_posix().lower(): img_path for img_path in image_paths}
                
                # Add to dictionary with subfolder name as key and list of image paths as value
                images_dict[subfolder.name] = list(unique_image_paths.values())
        
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
        Store results.
        """
        if os.path.isfile(self.results_object) and self.cache:
            print("Recovering results from cache")
            self.result_df = pickle.load(open(str(self.results_object), "rb"))
        else:
            processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
            model.to("cuda:0")

            results = []
            print("Iniciando llava")
            
            for cluster_name, image_paths in self.images_dict_format.items():
                print(f"Cluster {cluster_name}. Imágenes: {len(image_paths)}")
                for image_path in image_paths:
                    image = Image.open(image_path)
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
                    
                    start_time = time.time()
                    output = model.generate(**inputs, max_new_tokens=100)
                    classification_result = processor.decode(output[0], skip_special_tokens=True)
                    classification_category = classification_result.split(":")[-1].strip()
                    inference_time = time.time() - start_time

                    results.append({
                        "cluster": cluster_name,
                        "img": image_path,
                        "category_llava": classification_category,
                        "inference_time": inference_time
                    })

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_dir / "inference_results.csv", index=False, sep=";")
            pickle.dump(results_df, open(self.results_object, "wb"))
            self.result_df = results_df
            #logger.info(f"Results saved to {results_path}")


    def create_results_stats(self):
        """
        Generate statistics for each cluster's category distribution.
        Calculate the percentage of images in each cluster belonging to the same category.
        """
        category_counts = self.result_df.groupby(['cluster', 'category_llava']).size().reset_index(name='count')

        cluster_stats = []
        for cluster_name, group in category_counts.groupby('cluster'):
            total_images = group['count'].sum()
            predominant_category = group.loc[group['count'].idxmax(), 'category_llava']
            predominant_count = group['count'].max()
            success_percent = (predominant_count / total_images) * 100

            cluster_stats.append({
                'cluster': cluster_name,
                'total_img': total_images,
                'predominant_category': predominant_category,
                'success_percent': success_percent
            })

        self.result_stats_df = pd.DataFrame(cluster_stats)
        self.result_stats_df = self.result_stats_df.merge(category_counts, on='cluster', how='left')
        self.result_stats_df.to_csv(self.results_dir / "result_stats.csv", index=False, sep=";")
        
        self.plot_cluster_categories()






    def plot_cluster_categories(self, threshold=0.75):
        """
        Plot four stacked bar charts showing the category distribution within each cluster,
        excluding noise (cluster -1). Additionally, create a pie chart for the noise cluster.
        """
        # Exclude the noise (-1) from the main DataFrame and sort clusters numerically
        plot_data = self.result_stats_df[self.result_stats_df['cluster'] != '-1']
        plot_data['cluster_int'] = plot_data['cluster'].astype(int)  # Add helper column for sorting
        plot_data = plot_data.sort_values(by='cluster_int').reset_index(drop=True)
        plot_data['cluster'] = plot_data['cluster'].astype(str)  # Convert back to str for plotting
        plot_data = plot_data.drop(columns=['cluster_int'])  # Remove helper column

        total_images = self.result_stats_df['count'].sum()
        total_noise_images = self.result_stats_df[self.result_stats_df['cluster'] == '-1']['count'].sum()

        # Define a color map for each category based on all unique categories in result_stats_df
        unique_categories = self.result_stats_df['category_llava'].unique()
        colors = plt.cm.tab20c.colors  # A color map with enough colors
        category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}

        # Divide clusters into 4 groups for 4 charts
        n_clusters = plot_data['cluster'].nunique()
        clusters_per_plot = n_clusters // 4 + (1 if n_clusters % 4 != 0 else 0)
        clusters = plot_data['cluster'].unique()  # Ensures clusters are ordered

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(25, 15))
        fig.suptitle("Category Distribution by Cluster (Excluding Noise)\n" \
                     f"Total images: {total_images}", 
                     fontsize=16)

        for i in range(4):
            start_idx = i * clusters_per_plot
            end_idx = min((i + 1) * clusters_per_plot, n_clusters)
            subset_clusters = clusters[start_idx:end_idx]
            plot_subset = plot_data[plot_data['cluster'].isin(subset_clusters)]
            
            # Create stacked bar chart
            ax = axes[i // 2, i % 2]
            pivot_data = plot_subset.pivot_table(index='cluster', columns='category_llava', values='count', fill_value=0)
            pivot_data.plot(kind='bar', stacked=True, ax=ax, color=[category_colors[cat] for cat in pivot_data.columns])
            
            # Set green color for x-axis labels if success percent > 75%
            for label in ax.get_xticklabels():
                cluster_id = label.get_text()
                predominant_row = self.result_stats_df[(self.result_stats_df['cluster'] == cluster_id) &
                                                    (self.result_stats_df['category_llava'] == 
                                                        self.result_stats_df.loc[self.result_stats_df['cluster'] == cluster_id, 'predominant_category'].values[0])]
                success_percent = predominant_row['success_percent'].iloc[0]
                label.set_color('green' if success_percent > threshold*100 else 'black')
            
            # Title and individual plot labels
            ax.set_title(f"Clusters {subset_clusters[0]} to {subset_clusters[-1]}")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Image Count")
            ax.legend().set_visible(False)

        # Create the legend outside the loop
        fig.legend([plt.Line2D([0], [0], color=category_colors[cat], lw=4) for cat in unique_categories],
                unique_categories, title="Category", bbox_to_anchor=(0.92, 0.5), loc='center')
        plt.tight_layout(rect=[0, 0, 0.84, 0.95])
        fig.savefig(self.results_dir / "category_distribution_clusters.png")
        plt.close(fig)

        # Create pie chart for the noise cluster (-1)
        noise_data = self.result_stats_df[self.result_stats_df['cluster'] == '-1']
        if not noise_data.empty:
            noise_counts = noise_data.groupby('category_llava')['count'].sum()
            
            fig, ax = plt.subplots(figsize=(8, 10))  # Adjust height for better legend placement
            wedges, texts = ax.pie(noise_counts, startangle=90, colors=[category_colors[cat] for cat in noise_counts.index])
            
            # Add a detailed legend below the chart
            plt.legend(wedges, [f"{label}: {value:.1f}%" for label, value in zip(noise_counts.index, 
                    (noise_counts / noise_counts.sum() * 100).round(1))], 
                    title="Categories", loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=2)
            
            ax.set_title("Category Distribution in Noise Cluster (-1)" \
                        f"Total noise images: {total_noise_images}")
            fig.savefig(self.results_dir / "noise_cluster_pie_chart.png")
            plt.close(fig)





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

    llava = LlavaInference(images_dict_format=None, classification_lvl=3, experiment_name="hdbscan_optuna_2_dims_umap_072",cache=False)
    llava.run()
