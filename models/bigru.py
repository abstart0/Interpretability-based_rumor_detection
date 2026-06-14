"""
BiGRU (Bidirectional GRU) model for rumor binary classification.
"""

import torch
import torch.nn as nn


class BiGRU(nn.Module):
    """
    Bidirectional GRU with an embedding layer followed by a linear classifier.

    Architecture:
        Embedding → BiGRU → (concat last hidden states) → Linear → logits
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bigru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # BiGRU outputs hidden_dim*2 (concatenated forward + backward)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token id tensor, shape (batch_size, seq_len)

        Returns:
            logits: shape (batch_size,), raw scores before sigmoid
        """
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, emb_dim)
        emb = self.dropout(emb)
        _, h_n = self.bigru(emb)
        # h_n: (2, batch, hidden_dim)
        # concat forward and backward last hidden states
        h = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, hidden_dim*2)
        h = self.dropout(h)
        logits = self.fc(h).squeeze(1)  # (batch,)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary predictions (0 or 1)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = (torch.sigmoid(logits) > 0.5).long()
        return preds
