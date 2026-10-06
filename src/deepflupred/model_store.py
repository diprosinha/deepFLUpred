"""Load bundled or user-supplied deepFLUpred DNABERT+BiLSTM model bundles."""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import torch

from deepflupred.constants import PATHOGENICITY_HOSTS, SEGMENTS
from deepflupred.dnabert_bilstm import BiLSTMOnEmbeddings, LSTM_HIDDEN, N_WINDOWS, get_device, load_encoder
from deepflupred.oneclass import AvianInfluenzaScreen


class ModelStore:
    """Model repository supporting bundled resources and filesystem overrides.
    Also owns the single shared frozen DNABERT-2 encoder (expensive to load,
    so it's created lazily and cached for the lifetime of the ModelStore)."""

    def __init__(
        self,
        gene_model: str | Path | None = None,
        subtype_model_dir: str | Path | None = None,
        pathogenicity_model_dir: str | Path | None = None,
        oneclass_model_dir: str | Path | None = None,
    ) -> None:
        self.gene_model = Path(gene_model) if gene_model else None
        self.subtype_model_dir = Path(subtype_model_dir) if subtype_model_dir else None
        self.pathogenicity_model_dir = (
            Path(pathogenicity_model_dir) if pathogenicity_model_dir else None
        )
        self._gene_cache: dict | None = None
        self._subtype_cache: dict[str, dict] = {}
        self._pathogenicity_cache: dict[str, dict] = {}
        self._encoder_cache: tuple | None = None
        self.device = get_device()
        self.screen = AvianInfluenzaScreen(self.device, oneclass_model_dir)

    @property
    def encoder(self):
        """(tokenizer, frozen DNABERT-2 model) pair, loaded once and reused
        across every prediction stage."""
        if self._encoder_cache is None:
            self._encoder_cache = load_encoder(self.device)
        return self._encoder_cache

    def _build_from_state_dict(self, state_dict: dict, classes: list) -> BiLSTMOnEmbeddings:
        """Shape the head from the checkpoint rather than from constants, so a
        bundle retrained with a different hidden size still loads.
        `lstm.weight_ih_l0` is (4 * hidden, in_dim)."""
        in_dim = state_dict["lstm.weight_ih_l0"].shape[1]
        hidden = state_dict["lstm.weight_ih_l0"].shape[0] // 4
        model = BiLSTMOnEmbeddings(in_dim, hidden, len(classes))
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    def _load_bundle_dir(self, dir_path: Path) -> dict:
        state_dict = torch.load(dir_path / "model.pt", map_location="cpu")
        classes = json.loads((dir_path / "classes.json").read_text())
        metrics_path = dir_path / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        return {
            "model": self._build_from_state_dict(state_dict, classes),
            "classes": classes,
            "metrics": metrics,
        }

    def _load_bundle_resource(self, resource_dir) -> dict:
        state_dict = torch.load(
            (resource_dir / "model.pt").open("rb"), map_location="cpu", weights_only=True
        )
        classes = json.loads((resource_dir / "classes.json").read_text())
        metrics_resource = resource_dir / "metrics.json"
        metrics = json.loads(metrics_resource.read_text()) if metrics_resource.is_file() else None
        return {
            "model": self._build_from_state_dict(state_dict, classes),
            "classes": classes,
            "metrics": metrics,
        }

    def load_gene_model(self) -> dict:
        """Phase 2: the eight-class gene classifier."""
        if self._gene_cache is not None:
            return self._gene_cache
        if self.gene_model:
            self._gene_cache = self._load_bundle_dir(self.gene_model)
        else:
            resource = files("deepflupred") / "resources/models/gene_model"
            self._gene_cache = self._load_bundle_resource(resource)
        return self._gene_cache

    def load_subtype(self, segment: str) -> dict:
        segment = segment.upper()
        if segment not in SEGMENTS:
            raise ValueError(f"Unsupported segment: {segment}")
        if segment in self._subtype_cache:
            return self._subtype_cache[segment]
        if self.subtype_model_dir:
            bundle = self._load_bundle_dir(self.subtype_model_dir / segment)
        else:
            resource = files("deepflupred") / f"resources/models/subtype_models/{segment}"
            bundle = self._load_bundle_resource(resource)
        self._subtype_cache[segment] = bundle
        return bundle

    def load_pathogenicity(self, host: str) -> dict:
        host = host.lower()
        if host not in PATHOGENICITY_HOSTS:
            raise ValueError(f"Unsupported host: {host}. Choose from {PATHOGENICITY_HOSTS}")
        if host in self._pathogenicity_cache:
            return self._pathogenicity_cache[host]
        if self.pathogenicity_model_dir:
            bundle = self._load_bundle_dir(self.pathogenicity_model_dir / host)
        else:
            resource = files("deepflupred") / f"resources/models/pathogenicity_models/{host}"
            bundle = self._load_bundle_resource(resource)
        self._pathogenicity_cache[host] = bundle
        return bundle

    def inventory(self) -> list[dict]:
        rows = []
        meta = self.screen.metadata() or {}
        if meta:
            evaluation = meta.get("evaluation") or {}
            rows.append({
                "model": "oneclass/phase1",
                "task": meta.get("task", "phase 1 avian influenza A screen"),
                "classes": ["accepted", "rejected"],
                "test_metrics": {
                    key: value for key, value in evaluation.items()
                    if isinstance(value, (int, float)) and (
                        key.endswith("_rejected") or key.endswith("_auroc")
                        or key.endswith("_accepted") or key == "heldout_positive_acceptance"
                    )
                },
            })
        gene = self.load_gene_model()
        gene_metrics = gene.get("metrics") or {}
        rows.append({
            "model": "gene_model",
            "task": "phase 2 eight-class gene identification",
            "classes": gene.get("classes"),
            "test_metrics": {
                key: gene_metrics[key] for key in ("test_accuracy", "test_macro_f1")
                if key in gene_metrics
            } or gene_metrics,
        })
        for segment in SEGMENTS:
            bundle = self.load_subtype(segment)
            rows.append({
                "model": f"subtype_models/{segment}",
                "task": f"phase 3 {segment} subtype classification",
                "classes": bundle.get("classes"),
                "test_metrics": bundle.get("metrics"),
            })
        for host in PATHOGENICITY_HOSTS:
            bundle = self.load_pathogenicity(host)
            rows.append({
                "model": f"pathogenicity_models/{host}",
                "task": f"phase 4 HPAI/LPAI classification ({host} cleavage-site window)",
                "classes": bundle.get("classes"),
                "test_metrics": bundle.get("metrics"),
            })
        return rows

    @staticmethod
    def verify_bundled_models() -> dict[str, bool]:
        """Verify bundled model files against the packaged SHA-256 manifest."""
        model_root = files("deepflupred") / "resources/models"
        manifest = json.loads((model_root / "manifest.json").read_text(encoding="utf-8"))
        results = {}
        for relative_path, expected in manifest["sha256"].items():
            observed = hashlib.sha256((model_root / relative_path).read_bytes()).hexdigest()
            results[relative_path] = observed == expected
        return results
