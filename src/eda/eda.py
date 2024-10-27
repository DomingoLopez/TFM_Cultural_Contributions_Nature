
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
                 ):
        
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
        
        
    def __do_PCA(self, show_plots=False, dimensions=2):
        """
        PCA Dim reduction. 
        """
        logger.info("Using PCA Dim. reduction...")
        pca = PCA(n_components=dimensions)
        pca_result = pca.fit_transform(self.embeddings_df.values)
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

        return pca_df

    
    def __do_UMAP(self, show_plots=False, dimensions=2):
        """
        UMAP Dim reduction. 
        More info in https://umap-learn.readthedocs.io/en/latest/
        """
        logger.info("Using UMAP Dim. reduction")
        reducer = umap.UMAP(n_components=dimensions)
        umap_result = reducer.fit_transform(self.embeddings_df.values)
        umap_df = pd.DataFrame(data=umap_result)
        # Show only 2 dimensions in plots
        if show_plots:
            plt.scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.5)
            plt.title("Embeddings representation in 2D using UMAP")
            plt.show()
        
        return umap_df
        
        
    def __do_CVAE(self, dimensions=2, show_plots=True):
        """
        Compression VAE Dim reduction. 
        More info in https://github.com/maxfrenzel/CompressionVAE
        https://maxfrenzel.com/articles/compression-vae
        pip install tensorflow keras
        git clone https://github.com/maxfrenzel/CompressionVAE.git
        cd CompressionVAE
        pip install -e .
        """
        logger.info("Using CVAE Dim. reduction")
        # 1: Obtain array of embeddings
        X = self.embeddings_df.values
        # 2: Initialize cvae model with selected dimensions
        embedder = cvae.CompressionVAE(X, dim_latent=dimensions, verbose = False)
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

        return cvae_df



    def run_eda(self, dimensions=2, show_plots=True, dim_reduction ="cvae"):
        """
        Execute eda, including plots if selected and other stuff
        """
        self.__simple_eda(show_plots)
        embeddings_df = None
        # Apply dim reduction if chosen
        if dim_reduction is not None:
            if dim_reduction == "umap":
                embeddings_df = self.__do_UMAP(show_plots=show_plots, dimensions=dimensions)
            elif dim_reduction == "cvae":
                embeddings_df = self.__do_CVAE(show_plots=show_plots, dimensions=dimensions)
            elif dim_reduction == "pca":
                embeddings_df = self.__do_PCA(show_plots=show_plots, dimensions=dimensions)
            else:
                embeddings_df = self.__do_UMAP(show_plots=show_plots, dimensions=dimensions)
        else:
            embeddings_df = self.embeddings_df
            
        return embeddings_df
           
            

if __name__ == "__main__":
    # Objeto EDA.
    eda = EDA(embeddings=None, verbose=False)
    # devolemos los embeddings, con reducción o sin reducción según hayamos escogido
    embeddings_df = eda.run_eda(show_plots=False, dim_reduction="umap", dimensions=3)
    print(embeddings_df.shape)
    

