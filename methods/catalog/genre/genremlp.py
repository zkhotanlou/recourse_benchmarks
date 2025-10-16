"""
GenRe MLP: MLP architecture from GenRe paper (ICLR 2025).

This module implements the MLP classifier used in the GenRe paper:
- Architecture: 3 hidden layers with 10 neurons each, ReLU activation
- Training: Adam optimizer, lr=0.001, 100 epochs, batch_size=64

Reference:
    Garg et al. "From Search to Sampling: Generative Models for Robust 
    Algorithmic Recourse" ICLR 2025
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from models.api import MLModel


class GenReMLP(nn.Module):
    """
    MLP architecture from GenRe paper.
    
    Architecture:
        - 3 hidden layers with 10 neurons each
        - ReLU activation
        - Softmax output for 2 classes
    
    Args:
        n_features: Number of input features
    """
    
    def __init__(self, n_features):
        super(GenReMLP, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.softmax(x, dim=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input features (numpy array or torch tensor)
            
        Returns:
            numpy array of shape (n_samples, 2) with class probabilities
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            return self(X).numpy()


def train_genremlp(dataset, device='cpu', seed=42):
    """
    Train GenRe MLP following paper settings.
    
    Args:
        dataset: DataCatalog instance
        device: 'cpu' or 'cuda'
        seed: Random seed for reproducibility
        
    Returns:
        Trained GenReMLP model
    """
    # Fix random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Prepare data
    X_train = dataset.df_train.drop(columns=['y']).values
    y_train = dataset.df_train['y'].values
    
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    
    # Initialize model
    model = GenReMLP(n_features=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters (from paper)
    batch_size = 64
    epochs = 100
    dataset_size = len(X_train_tensor)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = torch.randperm(dataset_size)
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:min(i+batch_size, dataset_size)]
            batch_X = X_train_tensor[batch_indices]
            batch_y = y_train_tensor[batch_indices]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / n_batches
            print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}')
    
    model.eval()
    return model


def save_genremlp(model, dataset_name, save_dir='saved_models'):
    """
    Save trained GenRe MLP model.
    
    Args:
        model: Trained GenReMLP instance
        dataset_name: Name of dataset (e.g., 'compass', 'adult')
        save_dir: Directory to save model (relative to this file)
    """
    base_path = Path(__file__).parent
    save_path = base_path / save_dir / f'genremlp_{dataset_name}.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


def load_genremlp(n_features, dataset_name, save_dir='saved_models'):
    """
    Load trained GenRe MLP model.
    
    Args:
        n_features: Number of input features
        dataset_name: Name of dataset (e.g., 'compass', 'adult')
        save_dir: Directory containing saved model
        
    Returns:
        Loaded GenReMLP instance
    """
    base_path = Path(__file__).parent
    load_path = base_path / save_dir / f'genremlp_{dataset_name}.pth'
    
    if not load_path.exists():
        raise FileNotFoundError(f"Model not found: {load_path}")
    
    model = GenReMLP(n_features)
    model.load_state_dict(torch.load(load_path, map_location='cpu'))
    model.eval()
    print(f"Model loaded from: {load_path}")
    
    return model


class GenREMLPWrapper(MLModel):
    """
    Wrapper to make GenReMLP compatible with repo's MLModel interface.
    """
    
    def __init__(self, data, model):
        super().__init__(data)
        self._model = model
        self._feature_input_order = list(data.df_train.drop(columns=['y']).columns)
        self._backend = "pytorch"  # Add this
    
    @property
    def feature_input_order(self):
        """Return ordered list of feature names."""
        return self._feature_input_order
    
    @property
    def backend(self):  # Add this property
        """Return backend type."""
        return self._backend
    
    @property
    def raw_model(self):
        """Return the underlying GenReMLP model."""
        return self._model
    
    def predict(self, x):
        """One-dimensional prediction for output interval [0, 1]."""
        return self.predict_proba(x)[:, 1].reshape(-1, 1)
    
    def predict_proba(self, x):
        """Two-dimensional probability prediction."""
        if isinstance(x, pd.DataFrame):
            x = x[self._feature_input_order].values
        return self._model.predict_proba(x)


# Example usage
if __name__ == "__main__":
    from data.catalog import DataCatalog
    
    # Load dataset
    print("Loading COMPASS dataset...")
    dataset = DataCatalog("compass", "mlp", 0.7)
    
    # Train model
    print("\nTraining GenRe MLP...")
    model = train_genremlp(dataset, device='cpu')
    
    # Evaluate
    X_test = dataset.df_test.drop(columns=['y']).values
    y_test = dataset.df_test['y'].values
    
    proba = model.predict_proba(X_test)
    predicted = (proba[:, 1] > 0.5).astype(int)
    accuracy = (predicted == y_test).mean()
    
    print(f"\nGenRe MLP Accuracy: {accuracy:.2%}")
    print(f"Target (from paper): 69.60%")
    
    # Save model
    save_genremlp(model, 'compass')
    
    # Test wrapper
    wrapped_model = GenREMLPWrapper(dataset, model)
    print(f"\n✅ Model wrapped successfully!")
    print(f"Feature order: {wrapped_model.feature_input_order}")