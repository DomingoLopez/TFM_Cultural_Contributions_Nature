from pathlib import Path
import pickle
import pandas as pd
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import time
import os
import matplotlib.pyplot as plt

#from src.utils.image_loader import ImageLoader
#from loguru import logger


class LlavaInferenceRemote():
    """
    LlavaInferenceRemote for execution on remote, with only archives needed
    """
    def __init__(self, 
                 classification_lvl: str,
                 experiment:int,
                 name:str,
                 n_prompt:int,
                 cache: bool = True, 
                 verbose: bool = False):
        """
        Loads images from every cluster in order to do some inference on llava on ugr gpus
        Args:
            images_cluster_dict (dict)
            classification_lvl (str): Classification level to be used
            experiment_name (str): Name of the experiment for organizing results
        """
        self.experiment = experiment
        self.name = name
        self.base_dir = Path(__file__).resolve().parent / f"cluster_images/experiment_{experiment}" / f"{name}"
        self.results_dir = Path(__file__).resolve().parent / f"results/classification_lvl_{classification_lvl}/experiment_{experiment}" / f"{name}" / f"prompt_{n_prompt}"
        self.results_object = self.results_dir / f"result.pkl"
        self.results_object_next = self.results_dir / f"result_next.pkl"
        self.classification_lvls_dir = Path(__file__).resolve().parent / "classification_lvls/"
 
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Load categories based on classification level
        self.classification_lvl = classification_lvl
        self.categories = pd.read_csv(os.path.join(self.classification_lvls_dir, f"classification_level_{self.classification_lvl}.csv"), header=None, sep=";").iloc[:, 0].tolist()
        self.cache = cache

        # Initialize images_dict_format
        self.images_dict_format = self.get_cluster_images_dict()

        # Results
        self.result_df = None
        self.result_stats_df = None

        categories_joins = ", ".join([category.upper() for category in self.categories])
        self.prompt_1 = "You are an Image Classification Assistant especialized in cultural ecosystem services and cultural nature contributions to people." \
                        f"If the focus of the image is related to the general topic of cultural ecosystem services or cultural nature contributions to people, classify it as one of these {len(self.categories)} categories: {categories_joins}" + ". " \
                        "Otherwise, if the image is not related to the general topic of cultural ecosystem services or cultural nature contributions to people classify it as 'Not valid'. " \
                        "You need to EXCLUSIVELY provide the classification, not the reasoning."

        self.prompt_2 = "You are an Image Classification Assistant especialized in cultural ecosystem services and cultural nature contributions to people." \
                        f"If the focus of the image is related to the general topic of cultural ecosystem services or cultural nature contributions to people, classify it as one of these {len(self.categories)} categories: {categories_joins}" + ". " \
                        "Otherwise, if the image is not related to the general topic of cultural ecosystem services or cultural nature contributions to people classify it as 'Not valid'. " \
                        "If you classify the image as 'Not Valid' you must output the Classification, and also the reasoning of why you chose 'Not Valid' but between curly braces"
        
        
        if n_prompt > 2 or n_prompt < 1:
                raise ValueError("n_prompt must be 1 or 2")
            
        self.prompt = self.prompt_1 if n_prompt == 1 else self.prompt_2






    def show_prompts(self):
        print(self.prompt_1)
        print(self.prompt_2)



    def get_cluster_images_dict(self, knn=None):
        
        cluster_images_dict = {}
        for cluster_dir in self.base_dir.iterdir():
            if cluster_dir.is_dir():
                cluster_id = int(cluster_dir.name)
                cluster_images_dict[cluster_id] = [str(img_path) for img_path in cluster_dir.iterdir() if img_path.is_file()]
        return dict(sorted(cluster_images_dict.items()))




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
            print("Launching llava")
            
            for cluster_name, image_paths in self.images_dict_format.items():
                print(f"Cluster {cluster_name}. Imágenes: {len(image_paths)}")
                for image_path in image_paths:
                    image = Image.open(image_path)
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt},
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
                        "output": classification_result,
                        "inference_time": inference_time
                    })

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_dir / "inference_results.csv", index=False, sep=";")
            pickle.dump(results_df, open(self.results_object, "wb"))
            self.result_df = results_df
            #logger.info(f"Results saved to {results_path}")


    def run_next(self):
        """
        Run Llava-Next inference for every image in each subfolder of the base path.
        Store results.
        """
        if os.path.isfile(self.results_object) and self.cache:
            print("Recovering results from cache")
            self.result_df = pickle.load(open(str(self.results_object), "rb"))
        else:
            processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
            model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            model.to("cuda:0")
            model.config.pad_token_id = model.config.eos_token_id

            results = []
            print("Launching llava-next")
            for cluster_name, image_paths in self.images_dict_format.items():
                print(f"Cluster {cluster_name}. Images: {len(image_paths)}")
                for image_path in image_paths:
                    try:
                        image = Image.open(image_path).convert("RGB")  # Ensure compatibility with the model
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": self.prompt},
                                    {"type": "image", "image": image},  
                                ],
                            },
                        ]
                        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

                        start_time = time.time()
                        output = model.generate(**inputs, max_new_tokens=100)

                        classification_result = processor.decode(output[0], skip_special_tokens=True)
                        
                        if "[/INST]" in classification_result:
                            classification_category = classification_result.split("[/INST]")[-1].strip()
                        else:
                            classification_category = "Unknown"  # Handle unexpected output format

                        inference_time = time.time() - start_time

                        results.append({
                            "cluster": cluster_name,
                            "img": image_path,
                            "category_llava_next": classification_category,
                            "output": classification_result,
                            "inference_time": inference_time
                        })
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_dir / "inference_results_next.csv", index=False, sep=";")
            pickle.dump(results_df, open(self.results_object_next, "wb"))
            self.result_df = results_df




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






if __name__ == "__main__":
    llava = LlavaInferenceRemote(3,1,"index_18_silhouette_0.755",1,False,False)
    llava.run_next()
    llava2 = LlavaInferenceRemote(3,1,"index_18_silhouette_0.755",1,False,False)
    llava2.run()
    llava3 = LlavaInferenceRemote(3,1,"index_18_silhouette_0.755",2,False,False)
    llava3.run_next()
    llava4 = LlavaInferenceRemote(3,1,"index_18_silhouette_0.755",2,False,False)
    llava4.run()
