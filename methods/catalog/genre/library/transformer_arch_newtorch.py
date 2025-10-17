# recourse_benchmarks/methods/catalog/genre/transformer_arch.py
"""
GenRe Transformer architecture.

This file contains the Transformer model architecture used by GenRe.
Architecture is copied from train_transformer.py for consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class BinnedOutputLayer(nn.Module):
    """Binned output layer for discretized generation."""
    
    def __init__(self, d_model, n_features, n_bins=50):
        super(BinnedOutputLayer, self).__init__()
        
        self.n_features = n_features
        self.n_bins = n_bins
        
        self.feature_heads = nn.ModuleList([
            nn.Linear(d_model, n_bins) for _ in range(n_features)
        ])
        
        self.register_buffer('bin_centers', torch.linspace(0, 1, n_bins))
        
    def forward(self, x):
        outputs = []
        for head in self.feature_heads:
            logits = head(x)
            outputs.append(logits)
        return outputs
    
    def sample(self, x, temperature=1.0):
        outputs = self.forward(x)
        samples = []
        
        for logits in outputs:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            bin_indices = torch.multinomial(
                probs.view(-1, self.n_bins), 
                num_samples=1
            ).view(logits.shape[:-1])
            
            values = self.bin_centers[bin_indices]
            samples.append(values)
        
        return torch.stack(samples, dim=-1)


class GenReTransformer(nn.Module):
    """
    GenRe Transformer for counterfactual generation.
    
    This is the core generative model that learns R_θ(x+|x-).
    """
    
    def __init__(self, n_features, d_model=32, nhead=4, 
                 num_encoder_layers=8, num_decoder_layers=8,
                 dim_feedforward=128, dropout=0.1, n_bins=50):
        super(GenReTransformer, self).__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.n_bins = n_bins
        
        # Input embeddings
        self.src_embedding = nn.Linear(n_features, d_model)
        self.tgt_embedding = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = BinnedOutputLayer(d_model, n_features, n_bins)
        
    def forward(self, src, tgt, tgt_mask=None):
        """Forward pass."""
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        logits = self.output_layer(output)
        
        return logits
    
    def generate(self, src, temperature=1.0, max_len=None):
        """
        Generate counterfactual autoregressively.
        
        Args:
            src: Source (factual): (batch_size, n_features)
            temperature: Sampling temperature (higher = more diverse)
            max_len: Maximum generation length
            
        Returns:
            Generated counterfactual: (batch_size, n_features)
        """
        if max_len is None:
            max_len = self.n_features
        
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            device = src.device
            
            # Prepare source
            src = src.unsqueeze(1)
            
            # Start generation
            generated = torch.zeros(batch_size, 1, self.n_features).to(device)
            
            # Generate feature by feature
            for t in range(max_len):
                logits = self.forward(src, generated)
                
                # Sample next feature values
                next_values = []
                for feat_idx in range(self.n_features):
                    feat_logits = logits[feat_idx][:, -1, :]
                    
                    feat_logits = feat_logits / temperature
                    probs = F.softmax(feat_logits, dim=-1)
                    bin_idx = torch.multinomial(probs, num_samples=1)
                    
                    value = self.output_layer.bin_centers[bin_idx].squeeze(-1)
                    next_values.append(value)
                
                next_feat = torch.stack(next_values, dim=-1).unsqueeze(1)
                
                if t < max_len - 1:
                    generated = torch.cat([generated, next_feat], dim=1)
                else:
                    generated = next_feat
            
            return generated.squeeze(1)