import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
from loguru import logger
from tqdm import tqdm


class DinoV2Classifier:
    """
    A classifier that uses DinoV2 embeddings.
    Embeddings are either loaded from cache or generated on-the-fly.
    Includes methods for training, prediction, and accuracy evaluation.
    """
    def __init__(self, dinov2_inference, num_classes, cache_path=None):
        """
        Initialize the classifier with DinoV2Inference and classifier setup.
        Args:
            dinov2_inference: Instance of DinoV2Inference for generating embeddings.
            num_classes: Number of output classes for the classifier.
            cache_path: Path to save/load the embeddings cache.
        """
        self.dinov2_inference = dinov2_inference
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_path = cache_path or self.dinov2_inference.embeddings_cache_path
        
        # Initialize the classifier model
        embedding_dim = 768  # Adjust based on your DINOv2 model
        self.model = nn.Linear(embedding_dim, num_classes).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

    def load_or_generate_embeddings(self):
        """
        Load embeddings from cache or generate them if cache does not exist.
        """
        if self.cache_path and Path(self.cache_path).exists():
            logger.info(f"Loading embeddings from cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                embeddings_data = pickle.load(f)
        else:
            logger.info("Generating embeddings...")
            embeddings_data = self.dinov2_inference.run()
            if self.cache_path:
                logger.info(f"Saving embeddings to cache: {self.cache_path}")
                with open(self.cache_path, "wb") as f:
                    pickle.dump(embeddings_data, f)
        return embeddings_data

    def train(self, embeddings, labels, epochs=10, batch_size=32):
        """
        Train the classifier on provided embeddings and labels.
        Args:
            embeddings: Precomputed embeddings.
            labels: Corresponding labels for embeddings.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        dataset = EmbeddingDataset(embeddings, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for batch_embeddings, batch_labels in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
                batch_embeddings, batch_labels = batch_embeddings.to(self.device), batch_labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_embeddings)
                loss = self.criterion(outputs, batch_labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")

    def predict(self, embeddings):
        """
        Predict the classes for given embeddings.
        Args:
            embeddings: Precomputed embeddings to classify.
        Returns:
            List of predicted class indices.
        """
        self.model.eval()
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(embeddings_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def evaluate_accuracy(self, embeddings, labels):
        """
        Evaluate the accuracy of the classifier on provided embeddings and labels.
        Args:
            embeddings: Precomputed embeddings.
            labels: Ground truth labels.
        Returns:
            Accuracy score as a percentage.
        """
        predictions = self.predict(embeddings)
        accuracy = accuracy_score(labels, predictions)
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy * 100


class EmbeddingDataset(Dataset):
    """
    A PyTorch Dataset for embeddings and labels.
    """
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
