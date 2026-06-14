"""
BiGRU (Bidirectional GRU) model for rumor binary classification.

Architecture:
    Embedding → BiGRU → concat(h_f, h_b) → Dropout → Linear → logits
"""

import torch
import torch.nn as nn


class BiGRU(nn.Module):
    """
    Bidirectional GRU with output dropout for binary rumor classification.

    Architecture: Embedding → BiGRU → Dropout → Linear → logits

    Note: Embedding dropout is intentionally omitted — it cripples word
    representations at moderate dropout rates, reducing accuracy ~2-3%.

    Args:
        vocab_size:    size of input vocabulary
        embedding_dim: word embedding dimension (default 150)
        hidden_dim:    GRU hidden state dimension (default 200)
        num_layers:    number of GRU layers (default 1; more overfits small data)
        dropout:       dropout rate after GRU output (default 0.5)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.bigru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        gru_out_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token id tensor, shape (batch_size, seq_len)
        Returns:
            logits: shape (batch_size,), raw scores before sigmoid
        """
        emb = self.embedding(x)                         # (B, L, E)
        _, h_n = self.bigru(emb)                        # (2*D, B, H)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)       # (B, 2H): last layer fwd + bwd
        h = self.dropout(h)
        logits = self.fc(h).squeeze(1)                  # (B,)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary predictions (0 or 1)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = (torch.sigmoid(logits) > 0.5).long()
        return preds