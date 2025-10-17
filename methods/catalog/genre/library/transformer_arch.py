"""
GenRe Transformer architecture (Python 3.7 / PyTorch 1.x compatible).
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
        
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model) for old PyTorch
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0), :]
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
        """
        Args:
            x: (seq_len, batch, d_model)
        Returns:
            List of logits for each feature
        """
        outputs = []
        for head in self.feature_heads:
            logits = head(x)  # (seq_len, batch, n_bins)
            outputs.append(logits)
        return outputs


class GenReTransformer(nn.Module):
    """
    GenRe Transformer (compatible with Python 3.7 / old PyTorch).
    
    Note: Old PyTorch uses (seq_len, batch, features) format.
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
        
        # Transformer (old PyTorch - no batch_first)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = BinnedOutputLayer(d_model, n_features, n_bins)
        
    def forward(self, src, tgt, tgt_mask=None):
        """
        Forward pass.
        
        Args:
            src: (batch, seq_len, n_features) - will be transposed
            tgt: (batch, seq_len, n_features) - will be transposed
            
        Returns:
            List of logits for each feature
        """
        # Transpose: (batch, seq, feat) -> (seq, batch, feat)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        
        # Embed
        src = self.src_embedding(src)  # (seq, batch, d_model)
        tgt = self.tgt_embedding(tgt)  # (seq, batch, d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)  # (seq, batch, d_model)
        
        # Output layer
        logits = self.output_layer(output)
        
        # Transpose back: (seq, batch, bins) -> (batch, seq, bins)
        logits = [l.transpose(0, 1) for l in logits]
        
        return logits
    
    def generate(self, src, temperature=1.0, max_len=None):
        """
        Generate counterfactual autoregressively.
        
        Args:
            src: (batch, n_features)
            temperature: Sampling temperature
            max_len: Maximum generation length
            
        Returns:
            Generated counterfactual: (batch, n_features)
        """
        if max_len is None:
            max_len = self.n_features
        
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            device = src.device
            self.to(device)
            
            # Prepare source: (batch, 1, n_features)
            src = src.unsqueeze(1)
            
            # Start generation
            generated = torch.zeros(batch_size, 1, self.n_features).to(device)
            
            # Generate feature by feature
            for t in range(max_len):
                # Forward pass
                logits = self.forward(src, generated)
                
                # Sample next feature values
                next_values = []
                for feat_idx in range(self.n_features):
                    feat_logits = logits[feat_idx][:, -1, :]  # (batch, n_bins)
                    
                    # Apply temperature and sample
                    feat_logits = feat_logits / temperature
                    probs = F.softmax(feat_logits, dim=-1)
                    bin_idx = torch.multinomial(probs, num_samples=1)
                    
                    # Convert to continuous value
                    value = self.output_layer.bin_centers[bin_idx].squeeze(-1)
                    next_values.append(value)
                
                # Stack: (batch, n_features)
                next_feat = torch.stack(next_values, dim=-1).unsqueeze(1)
                
                if t < max_len - 1:
                    generated = torch.cat([generated, next_feat], dim=1)
                else:
                    generated = next_feat
            
            return generated.squeeze(1)