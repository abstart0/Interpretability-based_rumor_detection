"""
TextCNN (Kim 2014) for rumor binary classification.

Architecture:
    Embedding → multi-kernel Conv2d → ReLU → MaxPool → concat → Dropout → Linear → logit

Designed for short texts (tweets). Multiple filter sizes capture n-gram patterns
at different granularities — critical for rumor detection where specific phrases
(e.g. "breaking", "scientists confirm", "not a hoax") are strong indicators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN with multiple filter sizes and max-over-time pooling.

    Args:
        vocab_size:    vocabulary size
        embedding_dim: word embedding dimension
        filter_sizes:  list of n-gram sizes for convolution kernels
        n_filters:     number of filters per filter size
        dropout:       dropout rate after pooling
        pretrained:    optional pre-trained embedding weight matrix
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 200,
        filter_sizes: tuple = (2, 3, 4, 5),
        n_filters: int = 128,
        dropout: float = 0.4,
        pretrained: torch.Tensor = None,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Load pre-trained embeddings if provided
        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)
            self.embedding.weight.requires_grad = True  # fine-tune

        # Multiple parallel convolutions, one per filter size
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=n_filters,
                kernel_size=(fs, embedding_dim),
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token id tensor, shape (batch_size, seq_len)

        Returns:
            logits: shape (batch_size,), raw scores before sigmoid
        """
        # x: (B, L) → emb: (B, L, E) → add channel dim: (B, 1, L, E)
        emb = self.embedding(x).unsqueeze(1)

        # Each conv: (B, 1, L, E) → (B, F, L-fs+1, 1) → squeeze → (B, F, L')
        # Max-pool over time: (B, F, L') → (B, F)
        pooled = []
        for conv in self.convs:
            feat = F.relu(conv(emb)).squeeze(3)      # (B, F, L')
            feat = F.max_pool1d(feat, feat.size(2))   # (B, F, 1)
            pooled.append(feat.squeeze(2))             # (B, F)

        # Concat all filter sizes: (B, F * num_filter_sizes)
        h = torch.cat(pooled, dim=1)
        h = self.dropout(h)
        logits = self.fc(h).squeeze(1)                # (B,)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary predictions (0 or 1)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = (torch.sigmoid(logits) > 0.5).long()
        return preds
