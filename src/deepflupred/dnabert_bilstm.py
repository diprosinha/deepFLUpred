"""DNABERT -> BiLSTM feature pipeline shared by every deepFLUpred model stage.

  DNA sequence
       |
  DNABERT tokenizer          (DNABERT-2 k-mer/BPE tokenizer)
       |
  k-mer tokens
       |
  DNABERT Transformer        (zhihan1996/DNABERT-2-117M, frozen)
       |
  Hidden states               (one 768-dim vector per token)
       |
  Pooling                     (adaptive average-pool down to a fixed
       |                       N_WINDOWS-step sequence, so variable-length
       |                       inputs -- a 110nt cleavage-site window through
       |                       a 1800nt HA gene -- all produce the same shape)
  Numerical feature vector    ([N_WINDOWS, 768] per sequence)
       |
  BiLSTM classifier           (trained per stage: segment ID, HA subtype,
                                NA subtype, pathogenicity)

Matches gisaid_data/new_analysis/run_dnabert_bilstm.py exactly, so the
already-trained BiLSTM checkpoints load and run unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
MAX_TOKENS = 512
N_WINDOWS = 16
EMBED_BATCH = 16
LSTM_HIDDEN = 64


class BiLSTMOnEmbeddings(nn.Module):
    """BiLSTM over a fixed-length sequence of pooled DNABERT embeddings."""
    def __init__(self, in_dim: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        pooled = torch.cat([h[0], h[1]], dim=1)
        return self.head(pooled)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_encoder(device: torch.device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device).eval()
    return tok, model


@torch.no_grad()
def pooled_embed_sequences(seqs, tok, model, device, n_windows: int = N_WINDOWS,
                            batch_size: int = EMBED_BATCH):
    """DNABERT tokenizer -> k-mer tokens -> DNABERT Transformer -> hidden
    states -> adaptive-avg-pool to a fixed [n_windows, hidden] numerical
    feature-vector sequence, for every input sequence. Returns a
    (len(seqs), n_windows, hidden_size) numpy array."""
    import numpy as np
    hidden_size = model.config.hidden_size
    out = np.zeros((len(seqs), n_windows, hidden_size), dtype=np.float32)
    for i in range(0, len(seqs), batch_size):
        batch = [s.upper().replace("U", "T") for s in seqs[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_TOKENS).to(device)
        hidden = model(**enc)[0]  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        hidden = hidden * mask
        # adaptive_avg_pool1d on MPS requires divisible sizes, so pool on CPU.
        pooled = F.adaptive_avg_pool1d(hidden.transpose(1, 2).cpu(), n_windows).transpose(1, 2)
        out[i:i + batch_size] = pooled.numpy()
    return out


@torch.no_grad()
def predict_proba(model: BiLSTMOnEmbeddings, embedding, device: torch.device):
    """embedding: (n_windows, hidden_size) numpy array for one sequence."""
    x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    logits = model(x)
    return torch.softmax(logits, dim=1)[0].cpu().numpy()
