"""Phase 1 -- the one-class screen that runs before any classifier.

    "Does this sequence belong to any gene of avian influenza A?"

A one-class BiLSTM over the frozen DNABERT-2 embedding the later phases also
use, trained on avian influenza A only -- all eight segments
(PB2/PB1/PA/HA/NP/NA/MP/NS), positives only. Either a Deep SVDD head (distance
to a fixed center) or a reconstruction autoencoder, whichever won selection
during training. See ``gisaid_data/new_analysis/phase12/``.

The screen reports a score and the threshold it was compared against. The
threshold is a percentile of the *training* score distribution, chosen a priori
rather than tuned against outliers, so a decision near the boundary is
genuinely borderline rather than the product of a fitted cut-off.

Two limits are worth stating plainly:

* Rejection means "unlike avian influenza A", not "identified as something
  else". The screen cannot say what a rejected sequence actually is.
* It screens for influenza A *genes*, not for host. Human-host influenza A is
  genetically close to avian influenza A, so passing means "influenza A-like",
  not "confirmed avian origin".

This replaces the two separate gates used before 0.2.0 (a classical OneClassSVM
"is this influenza?" gate plus an "is this H or N?" BiLSTM gate). The second
gate existed only because the old segment classifier was a forced HA-vs-NA
binary with no way to say "neither"; the phase-2 eight-class gene classifier
answers that directly, so the gate is no longer needed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GateResult:
    """The screen's verdict for one sequence."""

    name: str
    accepted: bool
    score: float
    threshold: float
    method: str

    @property
    def margin(self) -> float:
        """Distance from the decision boundary; negative when rejected."""
        return self.score - self.threshold

    def as_row(self, prefix: str) -> dict:
        return {
            f"{prefix}_decision": "accepted" if self.accepted else "rejected",
            f"{prefix}_score": round(self.score, 6),
            f"{prefix}_threshold": round(self.threshold, 6),
            f"{prefix}_margin": round(self.margin, 6),
        }


class OneClassBiLSTM(nn.Module):
    """Deep SVDD head: BiLSTM -> bias-free projection.

    Bias terms are disabled because Deep SVDD's objective is otherwise
    trivially satisfiable by emitting a constant.
    """

    def __init__(self, in_dim: int, hidden: int = 64, rep_dim: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, rep_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(torch.cat([h[0], h[1]], dim=1))


class OneClassBiLSTMAutoencoder(nn.Module):
    """Reconstruction head: BiLSTM encoder -> latent -> BiLSTM decoder."""

    def __init__(self, in_dim: int, hidden: int = 64, latent_dim: int = 32, n_steps: int = 16) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.encoder = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.to_latent = nn.Linear(hidden * 2, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden, batch_first=True, bidirectional=True)
        self.to_output = nn.Linear(hidden * 2, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        latent = self.to_latent(torch.cat([h[0], h[1]], dim=1))
        decoded, _ = self.decoder(latent.unsqueeze(1).repeat(1, self.n_steps, 1))
        return self.to_output(decoded)


class AvianInfluenzaScreen:
    """Loads and applies the phase-1 one-class screen.

    The model is looked up in the package resources unless an override
    directory is supplied. If the bundled model is absent the screen reports
    itself as unavailable rather than raising, so a package built without it
    still runs the classifier phases.
    """

    MODEL_FILE = "phase1_oneclass_bilstm.pt"
    METADATA_FILE = "phase1_metadata.json"

    def __init__(
        self,
        device: torch.device,
        model_dir: str | Path | None = None,
        threshold: float | None = None,
    ) -> None:
        """
        `threshold` overrides the bundled decision threshold. The bundled value
        corresponds to a 95% nominal acceptance rate on the training
        distribution; raising it rejects more, lowering it accepts more. The
        trade-off at other operating points is tabulated in
        ``new_analysis/phase12/results/phase1/``.
        """
        self.device = device
        self.model_dir = Path(model_dir) if model_dir else None
        self.threshold_override = threshold
        self._bundle = None
        self._loaded = False

    # -- resource plumbing ----------------------------------------------------
    def _read_bytes(self, filename: str) -> bytes | None:
        if self.model_dir:
            path = self.model_dir / filename
            return path.read_bytes() if path.exists() else None
        resource = files("deepflupred") / f"resources/models/oneclass/{filename}"
        return resource.read_bytes() if resource.is_file() else None

    def _read_metadata(self, filename: str) -> dict:
        if self.model_dir:
            path = self.model_dir / filename
            return json.loads(path.read_text()) if path.exists() else {}
        resource = files("deepflupred") / f"resources/models/oneclass/{filename}"
        return json.loads(resource.read_text()) if resource.is_file() else {}

    @property
    def bundle(self):
        if not self._loaded:
            self._loaded = True
            payload = self._read_bytes(self.MODEL_FILE)
            if payload is not None:
                import io

                checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
                method = checkpoint.get("method", "autoencoder")
                if method == "deep_svdd":
                    model = OneClassBiLSTM(
                        checkpoint["in_dim"], checkpoint.get("hidden", 64), checkpoint["rep_dim"]
                    )
                else:
                    model = OneClassBiLSTMAutoencoder(
                        checkpoint["in_dim"],
                        checkpoint.get("hidden", 64),
                        checkpoint["latent_dim"],
                        checkpoint.get("n_steps", 16),
                    )
                model.load_state_dict(checkpoint["state_dict"])
                model.to(self.device).eval()
                center = checkpoint.get("center")
                self._bundle = {
                    "model": model,
                    "method": method,
                    "threshold": float(checkpoint["threshold"]),
                    "center": center.to(self.device) if center is not None else None,
                }
        return self._bundle

    @torch.no_grad()
    def screen(self, embedding: np.ndarray) -> GateResult | None:
        """embedding: (16, 768) for one sequence. None if no model is bundled."""
        bundle = self.bundle
        if bundle is None:
            return None
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        if bundle["method"] == "deep_svdd":
            distance = torch.sum((bundle["model"](x) - bundle["center"]) ** 2, dim=1)
            score = -float(distance[0].cpu())
        else:
            error = torch.mean((bundle["model"](x) - x) ** 2, dim=(1, 2))
            score = -float(error[0].cpu())
        threshold = (
            float(self.threshold_override)
            if self.threshold_override is not None
            else bundle["threshold"]
        )
        return GateResult(
            name="avian_influenza",
            accepted=score >= threshold,
            score=score,
            threshold=threshold,
            method=str(bundle["method"]),
        )

    # -- reporting ------------------------------------------------------------
    def available(self) -> bool:
        return self.bundle is not None

    def metadata(self) -> dict:
        return self._read_metadata(self.METADATA_FILE)
