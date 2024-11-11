
from loguru import logger
import numpy as np
import sys
from pathlib import Path
import os
from PIL import Image
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import umap
from cvae import cvae



class EDA:
    """
    Exploratory Data Analysis, Visualization and Dimensionality reduction.
    There are so many parameters in Dim. reduction, specially in cvae.
    We could try some OPTUNA for hiperparameter tuning. Maybe in v2.
    """
    def __init__(self, 
                 embeddings=None,
                 verbose=False,
                 cache = True,
                 ):
        

        self.cache = cache
        # Setup logging
        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        # Take embeddings from cache if they dont exists
        if embeddings == None:
            cache_root_folder = Path(__file__).resolve().parent.parent.parent / "cache"  
            if Path(cache_root_folder).is_dir():
                logger.info("Accessing cache to recover latest embeddings generated.")
                files = [f for f in Path(cache_root_folder).glob('*') if f.is_file()]
    
            if not files:
                raise FileNotFoundError(f"El directorio de caché no contiene embeddings generados. No es posible recuperarlos. Indique embeddings a analizar.")
            
            # Obtener último archivo generado de embeddings
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            self.embeddings = pickle.load(
                open(str(latest_file), "rb")
            )
        else:
            self.embeddings = embeddings
            
        self.embeddings_df = pd.DataFrame(self.embeddings)

        # Dirs and files
        self.cache_dir = Path(__file__).resolve().parent / "cache" 
        os.makedirs(self.cache_dir, exist_ok=True)



    def __simple_eda(self, show_plots=False):
        """
        Generate simple eda. No plots in this.
        """
        logger.info("Showing Simple EDA info from embeddings")
        print(f"Embeddings dataframe shape: \n {self.embeddings_df.shape}\n")
        print(f"Embeddings dataframe head :\n {self.embeddings_df.head()}\n")
        print(f"Embeddings estatistics :\n {self.embeddings_df.describe()}\n")
        # Plot simple histogram
        if show_plots:
            plt.hist(self.embeddings_df.values.flatten(), bins=50)
            plt.title("Embeddings distribution")
            plt.show()

    
    def run_scaler(self, type="standard"):
        """
        Apply a specified scaler to the embeddings DataFrame.

        Parameters
        ----------
        type : str
            The type of scaler to apply. Options are "standard", "minmax", "robust", or "maxabs".
        
        Returns
        -------
        embeddings_scaled : pd.DataFrame
            The scaled embeddings as a DataFrame.
        """
        # If cache true and file exists, load embeddings scaled from cache
        embeddings_scaled_path = os.path.join(self.cache_dir, f"scaled/{type}.pkl")
        # Create folder if it doesnt exist
        os.makedirs(os.path.join(self.cache_dir,"scaled"), exist_ok=True)
        if self.cache and os.path.isfile(embeddings_scaled_path):
            try:
                embeddings_scaled = pickle.load(
                    open(str(embeddings_scaled_path), "rb")
                )
            except:
                FileNotFoundError("Couldnt find provided file with scaled embeddings")

            logger.info(f"Retrieving {type} scaler embeddings from cache")
            return embeddings_scaled
        
        else:
            # Scaler options
            scalers = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
                "maxabs": MaxAbsScaler()
            }

            # Get Scaler
            scaler = scalers.get(type, StandardScaler())
            logger.info(f"Applying {type} scaler to embeddings")

            # Apply scaler
            embeddings = scaler.fit_transform(self.embeddings_df.values)

            # return embeddings_scaled
            embeddings_scaled = pd.DataFrame(embeddings, columns=self.embeddings_df.columns)
            logger.debug(f"Embeddings scaled using {type.capitalize()} Scaler.")
            # Save embeddings_scaled
            pickle.dump(
                embeddings_scaled,
                open(str(embeddings_scaled_path), "wb")
            )
            return embeddings_scaled
            
        
    def __do_PCA(self, embeddings_df, scaler="standard", show_plots=True , reduction_params=None):
        """
        PCA Dim reduction. 
        """

        if reduction_params is None:
            reduction_params = {}
        # Check if they are available in cache
        embeddings_dim_red_df = self.check_reduced_exists_cache(scaler, "pca", dimensions)
        if embeddings_dim_red_df is None:
            logger.info(f"Using PCA Dim. reduction. {dimensions=}")
            pca = PCA(random_state=42, **reduction_params)
            pca_result = pca.fit_transform(embeddings_df.values)
            pca_df = pd.DataFrame(data=pca_result)
            # Eigenvectors
            eigenvectors = pca.components_
            print("Principal components (Eigenvectors):")
            print(eigenvectors)
            # Eigenvalues
            eigenvalues = pca.explained_variance_ratio_ 
            print("Explained variance ratio (Eigenvalues):")
            print(eigenvalues)
            # Show only 2 dimensions in plots
            if show_plots:
                plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
                plt.title("Embeddings representation in 2D using PCA")
                plt.show()

            # Save to cache
            self.__save_reduced_embeddings(pca_df, scaler, "pca", dimensions)
            return pca_df
        else:
            logger.info(f"Retrieving pca reduced embeddings from cache")
            return embeddings_dim_red_df

    



    def __do_UMAP(self, embeddings_df, scaler="standard", show_plots=True , reduction_params=None):
        """
        UMAP Dim reduction. 
        More info in https://umap-learn.readthedocs.io/en/latest/
        """
        if reduction_params is None:
            reduction_params = {}
        # Check if they are available in cache
        embeddings_dim_red_df = self.check_reduced_exists_cache(scaler, "umap", dimensions)
        if embeddings_dim_red_df is None:
            logger.info(f"Using UMAP Dim. reduction. {dimensions=}")
            reducer = umap.UMAP(random_state=42, **reduction_params)
            umap_result = reducer.fit_transform(embeddings_df.values)
            umap_df = pd.DataFrame(data=umap_result)
            # Show only 2 dimensions in plots
            if show_plots:
                plt.scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.5)
                plt.title("Embeddings representation in 2D using UMAP")
                plt.show()
            
            # Save to cache
            self.__save_reduced_embeddings(umap_df, scaler, "umap", dimensions)
            return umap_df
        else:
            logger.info(f"Retrieving umap reduced embeddings from cache")
            return embeddings_dim_red_df
        
        


    def __do_CVAE(self, embeddings_df, scaler="standard", show_plots=True , reduction_params=None):
        """
        Compression VAE Dim reduction. 
        More info in https://github.com/maxfrenzel/CompressionVAE
        https://maxfrenzel.com/articles/compression-vae
        pip install tensorflow keras
        git clone https://github.com/maxfrenzel/CompressionVAE.git
        cd CompressionVAE
        pip install -e .
        """
        if reduction_params is None:
            reduction_params = {}
        # Check if they are available in cache
        embeddings_dim_red_df = self.check_reduced_exists_cache(scaler, "cvae", dimensions)
        if embeddings_dim_red_df is None:
            logger.info(f"Using CVAE Dim. reduction: {dimensions=}")
            # 1: Obtain array of embeddings
            X = embeddings_df.values
            # 2: Initialize cvae model with selected dimensions
            # embedder = cvae.CompressionVAE(X, dim_latent=dimensions, verbose = False)
            embedder = cvae.CompressionVAE(X, **reduction_params, verbose = False)
            # 3: Train cvae model
            embedder.train()  
            # 4: Get reduced embeddings
            embeddings_compressed = embedder.embed(X)  
            cvae_df = pd.DataFrame(data=embeddings_compressed)
            # Show only 2 dimensions in plots
            if show_plots:
                plt.scatter(embeddings_compressed[:, 0], embeddings_compressed[:, 1], alpha=0.5)
                plt.title("Embeddings in latent space (CVAE compression)")
                plt.xlabel("Latent dim 1")
                plt.ylabel("Latent dim 2")
                plt.show()

            # Save to cache
            self.__save_reduced_embeddings(cvae_df, scaler, "cvae", dimensions)
            return cvae_df
        else:
            logger.info(f"Retrieving cvae reduced embeddings from cache")
            return embeddings_dim_red_df



    def check_reduced_exists_cache(self, scaler, dim_reduction, reduction_params):
        """
        Check if reduced embeddings are available in cache and load them if they are.

        Parameters
        ----------
        scaler : str
            The type of scaler applied (e.g., "standard", "minmax").
        dim_reduction : str
            The dimensionality reduction technique (e.g., "umap").
        dimensions : int
            The number of dimensions after reduction.

        Returns
        -------
        embeddings_dim_red : pd.DataFrame or None
            The cached reduced embeddings as a DataFrame if available, else None.
        """
        # Define the path based on the presence of scaler and dimensionality reduction
        if scaler is not None and dim_reduction is not None:
            path = os.path.join(self.cache_dir, f"scaled_and_reduced/{scaler}/{dim_reduction}_{dimensions}.pkl")
        else:
            path = os.path.join(self.cache_dir, f"dim_reduced/{dim_reduction}_{dimensions}.pkl")

        # Check if the file exists and load it if available
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    embeddings_dim_red = pickle.load(f)
                return embeddings_dim_red
            except FileNotFoundError:
                logger.error("Couldn't find provided file with reduced embeddings.")
                return None
        else:
            return None  # Return None if the file doesn't exist
                

    def __save_reduced_embeddings(self, df, scaler, dim_reduction, reduction_params):
        """
        Save reduced embeddings to a cache file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the reduced embeddings.
        scaler : str
            The type of scaler applied (e.g., "standard", "minmax").
        dim_reduction : str
            The dimensionality reduction technique (e.g., "umap").
        dimensions : int
            The number of dimensions after reduction.
        """
        # Define the path based on scaler and dimensionality reduction technique
        if scaler is not None and dim_reduction is not None:
            path = os.path.join(self.cache_dir, f"scaled_and_reduced/{scaler}/{dim_reduction}_{dimensions}.pkl")
        else:
            path = os.path.join(self.cache_dir, f"dim_reduced/{dim_reduction}_{dimensions}.pkl")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save reduced embeddings to cache
        with open(path, "wb") as f:
            pickle.dump(df, f)
        logger.info(f"Reduced embeddings saved to {path}")

            

    def run_dim_red(self, embeddings_df, dim_reduction ="umap", scaler = "standard", show_plots=True, reduction_params=None):
        """
        Execute eda, including plots if selected and other stuff
        """
        # Apply dim reduction if chosen
        if dim_reduction is not None:
            if dim_reduction == "umap":
                embeddings_dim_red = self.__do_UMAP(embeddings_df, scaler= scaler, show_plots=show_plots, reduction_params=reduction_params)
            elif dim_reduction == "cvae":
                embeddings_dim_red = self.__do_CVAE(embeddings_df, scaler= scaler, show_plots=show_plots, reduction_params=reduction_params)
            elif dim_reduction == "pca":
                embeddings_dim_red = self.__do_PCA(embeddings_df, scaler= scaler, show_plots=show_plots, reduction_params=reduction_params)
            else:
                embeddings_dim_red = self.__do_UMAP(embeddings_df,scaler= scaler, show_plots=show_plots, reduction_params=reduction_params)
        else:
            embeddings_dim_red = embeddings_df
            
        return embeddings_dim_red
           
            

if __name__ == "__main__":
    # Objeto EDA.
    eda = EDA(embeddings=None, verbose=False)
    # devolemos los embeddings, con reducción o sin reducción según hayamos escogido
    embeddings_df = eda.run_eda(show_plots=False, dim_reduction="umap", dimensions=3)
    print(embeddings_df.shape)
    

