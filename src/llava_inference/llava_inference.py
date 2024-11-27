from pathlib import Path
import pickle
import shutil
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import time
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import entropy


#from src.utils.image_loader import ImageLoader
#from loguru import logger


class LlavaInference():
    """
    LlavaInference allows us to deploy selected Llava model (locally or in NGPU - UGR, but without automation yet)
    We start with Llava1.5-7b params. It can download model, and do some inference given some images and text prompt as inputs.
    """
    def __init__(self, 
                 images: list,
                 classification_lvl: str,
                 best_experiment: pd.DataFrame,
                 n_prompt:int,
                 type:str,
                 cache: bool = True, 
                 verbose: bool = False):
        """
        Loads images from every cluster in order to do some inference on llava on ugr gpus
        Args:
            images_cluster_dict (dict)
            classification_lvl (str): Classification level to be used
            experiment_name (str): Name of the experiment for organizing results
        """

        if(type not in ("llava", "llava_next")):
            raise ValueError("type must be one of llava or llava_next")

        self.best_experiment = best_experiment
        self.images = images
        self.type = type
        # Base dir for moving images from every cluster.
        self.base_dir = Path(__file__).resolve().parent / f"cluster_images/experiment_{self.best_experiment['id']}" / f"index_{self.best_experiment['original_index']}_silhouette_{self.best_experiment['score_w/o_penalty']:.3f}"
        self.results_dir = Path(__file__).resolve().parent / f"results/classification_lvl_{classification_lvl}/experiment_{self.best_experiment['id']}" / f"index_{self.best_experiment['original_index']}_silhouette_{self.best_experiment['score_w/o_penalty']:.3f}" / f"prompt_{n_prompt}"
        self.results_object = self.results_dir / f"result_{self.type}.pkl"
        self.results_csv = self.results_dir / f"inference_results_{self.type}.csv"
        self.classification_lvls_dir = Path(__file__).resolve().parent / "classification_lvls/"
        
        # Ensure directories exist
        
        # shutil.rmtree(self.base_dir, ignore_errors=True)
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
        self.prompt_1 = (
            "You are an Image Classification Assistant specialized in identifying cultural ecosystem services and cultural nature contributions to people. "
            f"Your task is to classify images into one of the following {len(self.categories)} categories: {categories_joins}. "
            "Under no circumstances should you provide a category that is not listed above."
            "Please provide ONLY the classification as your response, without any reasoning or additional details."
            )
        
        self.prompt_2 = (
            "You are an Image Classification Assistant specialized in identifying cultural ecosystem services and cultural nature contributions to people. "
            f"Your task is to classify images into one of the following {len(self.categories)} categories: {categories_joins}. "
            "If the image's focus does not pertain to cultural ecosystem services or cultural nature contributions to people, classify it as 'NOT VALID'. "
            "Under no circumstances should you provide a category that is not listed above. "
            "Please provide ONLY the classification as your response, without any reasoning or additional details."
            )
        
        if n_prompt > 2 or n_prompt < 1:
                raise ValueError("n_prompt must be 1 or 2")
            
        self.prompt = self.prompt_1 if n_prompt == 1 else self.prompt_2



    def show_prompts(self):
        print(self.prompt_1)
        print(self.prompt_2)



    def get_cluster_images_dict(self, knn=None):
        """
        Finds the k-nearest neighbors for each centroid of clusters among points that belong to the same cluster.
        Returns knn points for each cluster in dict format in case knn is not None

        Parameters
        ----------
        knn : int
            Number of nearest neighbors to find for each centroid

        Returns
        -------
        sorted_cluster_images_dict : dictionary with images per cluster (as key)
        """

        cluster_images_dict = {}
        labels = self.best_experiment['labels']

        if knn is not None:
            used_metric = "euclidean"
            
            for idx, centroid in enumerate(tqdm(self.best_experiment['centers'], desc="Processing cluster dirs (knn images selected)")):
                # Filter points based on label mask over embeddings
                cluster_points = self.best_experiment['embeddings'].values[labels == idx]
                cluster_images = [self.images[i] for i in range(len(self.images)) if labels[i] == idx]
                # Adjust neighbors, just in case
                n_neighbors_cluster = min(knn, len(cluster_points))
                
                nbrs = NearestNeighbors(n_neighbors=n_neighbors_cluster, metric=used_metric, algorithm='auto').fit(cluster_points)
                distances, indices = nbrs.kneighbors([centroid])
                closest_indices = indices.flatten()
                
                # Get images for each cluster
                cluster_images_dict[idx] = [cluster_images[i] for i in closest_indices]

            # Get noise (-1)
            cluster_images_dict[-1] = [self.images[i] for i in range(len(self.images)) if labels[i] == -1]
            
        else:
            for i, label in enumerate(tqdm(labels, desc="Processing cluster dirs")):
                if label not in cluster_images_dict:
                    cluster_images_dict[label] = []
                cluster_images_dict[label].append(self.images[i])
        
        # Sort dictionary
        sorted_cluster_images_dict = dict(sorted(cluster_images_dict.items()))
        return sorted_cluster_images_dict





    def create_cluster_dirs(self):
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
        self.__run_llava() if self.type == "llava" else self.__run_llava_next()




    def __run_llava(self):
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
            results_df.to_csv(self.results_csv, index=False, sep=";")
            pickle.dump(results_df, open(self.results_object, "wb"))
            self.result_df = results_df
            #logger.info(f"Results saved to {results_path}")




    def __run_llava_next(self):
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
                            "category_llava": classification_category,
                            "output": classification_result,
                            "inference_time": inference_time
                        })
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")

            results_df = pd.DataFrame(results)
            results_df.to_csv(self.results_csv, index=False, sep=";")
            pickle.dump(results_df, open(self.results_object, "wb"))
            self.result_df = results_df




    def create_results_stats(self):
        """
        Generate statistics for each cluster's category distribution.
        Calculate the percentage of images in each cluster belonging to the same category.
        Also, calculate homogeneity and entropy for each cluster.
        """
        # Contar categorías por clúster
        category_counts = self.result_df.groupby(['cluster', 'category_llava']).size().reset_index(name='count')

        # Calcular estadísticas por clúster
        cluster_stats = []
        for cluster_name, group in category_counts.groupby('cluster'):
            total_images = group['count'].sum()
            predominant_category = group.loc[group['count'].idxmax(), 'category_llava']
            predominant_count = group['count'].max()
            success_percent = (predominant_count / total_images) * 100

            # Calcular homogeneidad
            homogeneity_k = predominant_count / total_images

            # Calcular entropía
            label_counts = group['count'].values
            probabilities = label_counts / label_counts.sum()
            entropy_k = entropy(probabilities, base=2)

            # Guardar estadísticas del clúster
            cluster_stats.append({
                'cluster': cluster_name,
                'total_img': total_images,
                'predominant_category': predominant_category,
                'success_percent': success_percent,
                'homogeneity_k': homogeneity_k,
                'entropy_k': entropy_k
            })

        # Crear DataFrame con estadísticas por clúster
        self.result_stats_df = pd.DataFrame(cluster_stats)

        # Guardar las estadísticas en un archivo CSV
        self.result_stats_df.to_csv(self.results_dir / f"result_stats_{self.type}.csv", index=False, sep=";")

        # Calcular la métrica de calidad usando calculate_clustering_quality
        quality_results = self.calculate_clustering_quality(self.result_stats_df, alpha=1.0)

        # Mostrar resultados de la métrica de calidad
        print("Resultados de la Métrica de Calidad:")
        print(f"Homogeneidad Global: {quality_results['homogeneity_global']:.4f}")
        print(f"Penalización Global: {quality_results['penalization_global']:.4f}")
        print(f"Métrica de Calidad: {quality_results['quality_metric']:.4f}")

        # Llamar a la función de visualización si es necesario
        self.plot_cluster_categories()







    def plot_cluster_categories(self, threshold=0.75):
        """
        Plot four stacked bar charts showing the category distribution within each cluster,
        excluding noise (cluster -1). Additionally, create a pie chart for the noise cluster.
        """
        plot_data = self.result_stats_df[self.result_stats_df['cluster'].astype(str) != '-1'].copy()
        plot_data['cluster_int'] = pd.to_numeric(plot_data['cluster'], errors='coerce')
        plot_data['cluster_int'] = plot_data['cluster_int'].astype(int)
        plot_data = plot_data.sort_values(by='cluster_int').reset_index(drop=True)
        plot_data['cluster'] = plot_data['cluster_int'].astype(str)

 
        total_images = self.result_stats_df['count'].sum()
        total_noise_images = self.result_stats_df[self.result_stats_df['cluster'].astype(str) == '-1']['count'].sum()

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
            
            # Asegurarse de que los datos estén agrupados y únicos
            plot_subset = plot_subset.groupby(['cluster', 'category_llava'], as_index=False).sum()
            
            # Asegurarse de que los datos estén ordenados por cluster_int
            plot_subset = plot_subset.sort_values(by='cluster_int')

            # Crear tabla pivote
            pivot_data = plot_subset.pivot_table(
                index='cluster', 
                columns='category_llava', 
                values='count', 
                aggfunc='sum',  # Agrega los valores duplicados
                fill_value=0
            )
            
            # Reindexar para garantizar el orden correcto
            pivot_data = pivot_data.reindex(index=plot_subset['cluster'].unique())

            # Crear gráfico de barras apiladas
            ax = axes[i // 2, i % 2]
            pivot_data.plot(kind='bar', stacked=True, ax=ax, color=[category_colors[cat] for cat in pivot_data.columns])
            
            # Colorear etiquetas del eje x según el porcentaje de éxito
            for label in ax.get_xticklabels():
                cluster_id = label.get_text()
                predominant_row = self.result_stats_df[(self.result_stats_df['cluster'].astype(str) == cluster_id) &
                                                    (self.result_stats_df['category_llava'] == 
                                                        self.result_stats_df.loc[self.result_stats_df['cluster'].astype(str) == cluster_id, 'predominant_category'].values[0])]
                if not predominant_row.empty:
                    success_percent = predominant_row['success_percent'].iloc[0]
                    label.set_color('green' if success_percent > threshold * 100 else 'black')

            # Etiquetas y título del gráfico
            ax.set_title(f"Clusters {subset_clusters[0]} to {subset_clusters[-1]}")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Image Count")
            ax.legend().set_visible(False)

        # Create the legend outside the loop
        fig.legend([plt.Line2D([0], [0], color=category_colors[cat], lw=4) for cat in unique_categories],
                unique_categories, title="Category", bbox_to_anchor=(0.93, 0.5), loc='center')
        plt.tight_layout(rect=[0, 0, 0.84, 0.95])
        fig.savefig(self.results_dir / f"category_distribution_clusters_{self.type}.png")
        plt.close(fig)

        # Create pie chart for the noise cluster (-1)
        noise_data = self.result_stats_df[self.result_stats_df['cluster'].astype(str) == '-1']
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
            fig.savefig(self.results_dir / f"noise_cluster_pie_chart_{self.type}.png")
            plt.close(fig)




    def calculate_clustering_quality(self,result_stats_df, alpha=1.0):
        """
        Calculate the quality metric for clustering based on homogeneity and entropy.

        Args:
            result_stats_df (pd.DataFrame): DataFrame with statistics for each cluster. 
                Expected columns:
                    - 'cluster': Cluster ID.
                    - 'total_img': Total images in the cluster.
                    - 'homogeneity_k': Proportion of images in the dominant category.
                    - 'entropy_k': Entropy of label distribution in the cluster.
            alpha (float): Weight for the entropy penalty in the quality metric.

        Returns:
            dict: A dictionary with the calculated metrics:
                - 'homogeneity_global': Weighted average of cluster homogeneities.
                - 'penalization_global': Weighted average of cluster entropies.
                - 'quality_metric': Final clustering quality score.
        """
        # Ensure the required columns are present
        required_columns = ['total_img', 'homogeneity_k', 'entropy_k']
        for col in required_columns:
            if col not in result_stats_df.columns:
                raise ValueError(f"Missing required column: {col} in result_stats_df")

        # Calculate the total number of images across all clusters
        total_images = result_stats_df['total_img'].sum()

        # Calculate global homogeneity (weighted average of homogeneity_k)
        homogeneity_global = (result_stats_df['total_img'] * result_stats_df['homogeneity_k']).sum() / total_images

        # Calculate global penalty (weighted average of entropy_k)
        penalization_global = (result_stats_df['total_img'] * result_stats_df['entropy_k']).sum() / total_images

        # Combine metrics into the quality score
        quality_metric = homogeneity_global - alpha * penalization_global

        # Return the results as a dictionary
        return {
            'homogeneity_global': homogeneity_global,
            'penalization_global': penalization_global,
            'quality_metric': quality_metric
        }




if __name__ == "__main__":
    pass
