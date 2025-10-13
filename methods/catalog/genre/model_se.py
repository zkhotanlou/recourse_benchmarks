"""
GenRe: Generative Recourse method implementation.

This module implements GenRe (Generative Recourse), which uses a trained
Transformer model to generate counterfactual explanations through forward
sampling rather than gradient-based optimization.

Key differences from other methods:
- No gradient descent during inference
- Generates counterfactuals by sampling from learned distribution R_θ(x+|x-)
- Can generate diverse counterfactuals
- Trained offline, fast inference

Paper: "From Search to Sampling: Generative Models for Robust Algorithmic Recourse"
ICLR 2025
"""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from methods.api import RecourseMethod
from methods.processing import check_counterfactuals, merge_default_parameters
from models.api import MLModel

from .library.transformer_arch import GenReTransformer


class GenRe(RecourseMethod):
    """
    GenRe (Generative Recourse) implementation.
    
    GenRe learns a conditional generative model R_θ(x+|x-) that directly
    generates counterfactual explanations through forward sampling.
    
    Unlike gradient-based methods (DiCE, Wachter, etc.), GenRe:
    1. Trains a Transformer model offline
    2. At inference time, simply samples from the learned distribution
    3. No gradient descent or iterative optimization needed
    4. Fast and can generate diverse counterfactuals
    
    Parameters
    ----------
    mlmodel : MLModel
        Black-box classifier from recourse_benchmarks.
        Note: GenRe was trained with its own classifier, but uses
        repo's classifier for validation to ensure fair comparison.
        
    hyperparams : dict
        Dictionary containing:
        - model_path: Path to trained GenRe Transformer
        - temperature: Sampling temperature (default: 1.0)
        - n_samples: Number of candidates to generate (default: 10)
        - sigma: Noise parameter (default: 0.0)
    
    Examples
    --------
    >>> from data.catalog import DataCatalog
    >>> from models.catalog import ModelCatalog
    >>> from methods.catalog.genre import GenRe
    >>> 
    >>> dataset = DataCatalog("compass", "mlp", 0.7)
    >>> mlmodel = ModelCatalog(dataset, "mlp", "pytorch")
    >>> 
    >>> genre = GenRe(mlmodel, hyperparams={
    ...     "model_path": "./genre_transformer.pth",
    ...     "temperature": 1.0,
    ...     "n_samples": 10
    ... })
    >>> 
    >>> factuals = dataset.df_test.head(10)
    >>> counterfactuals = genre.get_counterfactuals(factuals)
    
    References
    ----------
    Garg, P., Nagalapatti, L., & Sarawagi, S. (2025).
    From Search to Sampling: Generative Models for Robust Algorithmic Recourse.
    ICLR 2025.
    """
    
    _DEFAULT_HYPERPARAMS = {
        "data_name": None,
        "model_path": None,  # Required: path to trained Transformer
        "temperature": 1.0,  # Sampling temperature (higher = more diverse)
        "n_samples": 10,     # Number of candidate counterfactuals to generate
        "sigma": 0.0,        # Noise parameter (not used in current implementation)
    }
    
    def __init__(self, mlmodel: MLModel, hyperparams: Dict = None) -> None:
        """
        Initialize GenRe recourse method.
        
        Args:
            mlmodel: Black-box ML model from recourse_benchmarks
            hyperparams: Hyperparameter dictionary
            
        Raises:
            ValueError: If backend is not PyTorch
            ValueError: If model_path is not specified
            FileNotFoundError: If model file doesn't exist
        """
        # Check backend
        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} not in supported backends {supported_backends}. "
                f"GenRe requires PyTorch."
            )
        
        super().__init__(mlmodel)
        
        # Merge hyperparameters
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)
        
        # Validate model path
        if self._params["model_path"] is None:
            raise ValueError(
                "model_path must be specified in hyperparams. "
                "Please provide the path to the trained GenRe Transformer."
            )
        
        # Load trained Transformer
        self._transformer = self._load_transformer(self._params["model_path"])
        
        # Store device
        self._device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._transformer = self._transformer.to(self._device)
        self._transformer.eval()

    
    def _load_transformer(self, model_path: str) -> GenReTransformer:
        """
        Load trained GenRe Transformer.
        
        Args:
            model_path: Path to model checkpoint (relative to genre/ directory)
            
        Returns:
            Loaded GenRe Transformer model
        """
        # Construct absolute path ⭐
        # model_path is relative to methods/catalog/genre/
        base_path = Path(__file__).parent
        
        # If path doesn't start with "saved_models/", prepend it
        if not model_path.startswith("saved_models/"):
            model_path = f"saved_models/{model_path}"
        
        full_path = base_path / model_path
        
        if not full_path.exists():
            raise FileNotFoundError(
                f"GenRe Transformer model not found at: {full_path}\n"
                f"Expected location: methods/catalog/genre/{model_path}\n"
                f"Please ensure the model file exists."
            )
        
        # Load checkpoint
        checkpoint = torch.load(full_path, map_location='cpu')
        # for newer PyTorch versions, use:
        # checkpoint = torch.load(full_path, map_location='cpu', weights_only=False)
        
        # Initialize model
        model = GenReTransformer(
            n_features=checkpoint['n_features'],
            d_model=checkpoint['d_model'],
            nhead=checkpoint['nhead'],
            num_encoder_layers=checkpoint['num_encoder_layers'],
            num_decoder_layers=checkpoint['num_decoder_layers'],
            dim_feedforward=checkpoint['dim_feedforward'],
            dropout=checkpoint['dropout'],
            n_bins=checkpoint['n_bins']
        )
        
        # Load state dict with dimension conversion
        state_dict = checkpoint['model_state_dict']
    
        # Convert pos_encoder.pe from (1, seq, d_model) to (seq, 1, d_model)
        if 'pos_encoder.pe' in state_dict:
            pe = state_dict['pos_encoder.pe']  # (1, 100, 32)
            state_dict['pos_encoder.pe'] = pe.transpose(0, 1)  # (100, 1, 32)
    
        model.load_state_dict(state_dict)
    
        return model
    
    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate counterfactual explanations using GenRe.
        
        Process:
        1. For each factual instance:
           a. Generate N candidate counterfactuals using Transformer
           b. Evaluate candidates using repo's classifier
           c. Select the best candidate (highest validity score)
        2. Validate and format results
        
        Args:
            factuals: DataFrame of instances needing recourse
            
        Returns:
            DataFrame of counterfactual explanations
            
        Note:
            This method uses the repo's classifier (self._mlmodel) for
            validation, ensuring fair comparison with other methods.
        """
        # Ensure correct feature order
        factuals = self._mlmodel.get_ordered_features(factuals)
        
        # Convert to numpy
        factuals_np = factuals.values
        
        # Generate counterfactuals
        counterfactuals_list = []
        
        for i in range(len(factuals_np)):
            factual = factuals_np[i:i+1]  # Keep 2D shape
            
            # Generate N candidates using Transformer
            candidates = self._generate_candidates(
                factual,
                n_samples=self._params["n_samples"],
                temperature=self._params["temperature"]
            )
            
            # Select best candidate using repo's classifier
            best_cf = self._select_best_candidate(candidates, factual)
            
            counterfactuals_list.append(best_cf)
        
        # Convert to DataFrame
        df_cfs = pd.DataFrame(
            np.vstack(counterfactuals_list),
            columns=self._mlmodel.feature_input_order,
            index=factuals.index.copy()
        )
        
        # Validate using repo's standard validation
        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        
        return df_cfs
    
    def _generate_candidates(self, factual: np.ndarray, n_samples: int, 
                           temperature: float) -> np.ndarray:
        """
        Generate N candidate counterfactuals using Transformer.
        
        Args:
            factual: Single factual instance (1, n_features)
            n_samples: Number of candidates to generate
            temperature: Sampling temperature
            
        Returns:
            Candidate counterfactuals (n_samples, n_features)
        """
        # Convert to tensor
        factual_tensor = torch.from_numpy(factual).float().to(self._device)
        
        # Repeat for batch generation
        factual_batch = factual_tensor.repeat(n_samples, 1)
        
        # Generate using Transformer
        with torch.no_grad():
            candidates = self._transformer.generate(
                factual_batch,
                temperature=temperature
            )
        
        # Convert back to numpy
        candidates_np = candidates.cpu().numpy()
        
        # Clip to [0, 1] range (should already be, but just in case)
        candidates_np = np.clip(candidates_np, 0.0, 1.0)
        
        return candidates_np
    
    def _select_best_candidate(self, candidates: np.ndarray, 
                              factual: np.ndarray) -> np.ndarray:
        """
        Select best candidate using repo's classifier.
        
        Selection criteria:
        1. Validity: Highest predicted probability for positive class
        2. If multiple candidates are valid, choose closest to factual
        
        Args:
            candidates: Candidate counterfactuals (n_samples, n_features)
            factual: Original factual instance (1, n_features)
            
        Returns:
            Best candidate (n_features,)
        """
        # Convert to DataFrame for classifier
        candidates_df = pd.DataFrame(
            candidates,
            columns=self._mlmodel.feature_input_order
        )
        
        # Get validity scores using repo's classifier
        validity_scores = self._mlmodel.predict_proba(candidates_df)[:, 1]
        
        # Find candidates with validity > 0.5
        valid_mask = validity_scores > 0.5
        
        if valid_mask.any():
            # If valid candidates exist, choose closest to factual
            valid_candidates = candidates[valid_mask]
            valid_scores = validity_scores[valid_mask]
            
            # Compute distances to factual
            distances = np.linalg.norm(valid_candidates - factual, axis=1)
            
            # Choose valid candidate with smallest distance
            best_idx = np.argmin(distances)
            best_candidate = valid_candidates[best_idx]
        else:
            # If no valid candidates, choose one with highest validity score
            best_idx = np.argmax(validity_scores)
            best_candidate = candidates[best_idx]
        
        return best_candidate