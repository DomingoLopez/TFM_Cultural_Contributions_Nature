
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
    Exploratory Data Analysis and Visualization
    """
    def __init__(self, 
                 embeddings=None,
                 show_plots=True,
                 dim_reduction="cvae",
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
        self.show_plots = show_plots
        self.dim_reduction = dim_reduction


    def __simple_eda(self):
        """
        Generate simple eda. No plots in this.
        """
        logger.info("Showing Simple EDA info from embeddings")
        print(f"Embeddings dataframe shape: \n {self.embeddings_df.shape}\n")
        print(f"Embeddings dataframe head :\n {self.embeddings_df.head()}\n")
        print(f"Embeddings estatistics :\n {self.embeddings_df.describe()}\n")
        
        
    def __do_PCA(self, dimensions=2):
        """
        PCA Dim reduction. 
        """
        logger.info("Using PCA Dim. reduction...")
        pca = PCA(n_components=dimensions)
        pca_result = pca.fit_transform(self.embeddings_df.values)
        # Eigenvectors
        eigenvectors = pca.components_
        print("Principal components (Eigenvectors):")
        print(eigenvectors)
        # Eigenvalues
        eigenvalues = pca.explained_variance_ratio_ 
        print("Explained variance ratio (Eigenvalues):")
        print(eigenvalues)
        
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.title("Embeddings representation in 2D using PCA")
        plt.show()
    
    
    def __do_UMAP(self, dimensions=2):
        """
        UMAP Dim reduction. 
        More info in https://umap-learn.readthedocs.io/en/latest/
        """
        logger.info("Using UMAP Dim. reduction")
        reducer = umap.UMAP(n_components=dimensions)
        umap_result = reducer.fit_transform(self.embeddings_df.values)
        plt.scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.5)
        plt.title("Embeddings representation in 2D using UMAP")
        plt.show()
        
        
        
    def __do_CVAE(self, dimensions=2):
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
        
        X = self.embeddings_df.values
        # Paso 2: Inicializar el modelo CompressionVAE
        embedder = cvae.CompressionVAE(X, dim_latent=dimensions)
        # Paso 3: Entrenar el modelo
        embedder.train()  # Entrenar el modelo  
        # Paso 4: Obtener los embeddings reducidos
        embeddings_compressed = embedder.embed(X)  # Embeddings en el espacio latente de 2 dimensiones

        if embeddings_compressed.shape[1] == 2:
            plt.scatter(embeddings_compressed[:, 0], embeddings_compressed[:, 1], alpha=0.5)
            plt.title("Embeddings in latent space (CVAE compression)")
            plt.xlabel("Latent dim 1")
            plt.ylabel("Latent dim 2")
            plt.show()
        else:
            logger.warning("Embeddings dimensionality is not 2D, skipping visualization.")


    def run_eda(self):
        """
        Execute eda, including plots if selected and other stuff
        """
        self.__simple_eda()
        # Apply dim reduction
        match self.dim_reduction:
            case "umap":
                self.__do_UMAP()
            case "cvae":
                self.__do_CVAE()
            case "pca":
                self.__do_PCA()
           
            

if __name__ == "__main__":
    eda = EDA()
    eda.run_eda()
    

