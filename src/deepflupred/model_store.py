"""Load bundled or user-supplied deepFLUpred model bundles."""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import joblib

from deepflupred.constants import PATHOGENICITY_HOSTS, SEGMENTS


class ModelStore:
    """Model repository supporting bundled resources and filesystem overrides."""

    def __init__(
        self,
        segment_model: str | Path | None = None,
        subtype_model_dir: str | Path | None = None,
        pathogenicity_model_dir: str | Path | None = None,
    ) -> None:
        self.segment_model = Path(segment_model) if segment_model else None
        self.subtype_model_dir = Path(subtype_model_dir) if subtype_model_dir else None
        self.pathogenicity_model_dir = (
            Path(pathogenicity_model_dir) if pathogenicity_model_dir else None
        )
        self._segment_cache: dict | None = None
        self._subtype_cache: dict[str, dict] = {}
        self._pathogenicity_cache: dict[str, dict] = {}

    @staticmethod
    def _load_resource(resource) -> dict:
        with resource.open("rb") as handle:
            return joblib.load(handle)

    @staticmethod
    def _load_path(path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    def load_segment_model(self) -> dict:
        if self._segment_cache is not None:
            return self._segment_cache
        if self.segment_model:
            self._segment_cache = self._load_path(self.segment_model)
        else:
            resource = files("deepflupred") / "resources/models/segment_model/model.joblib"
            self._segment_cache = self._load_resource(resource)
        return self._segment_cache

    def load_subtype(self, segment: str) -> dict:
        segment = segment.upper()
        if segment not in SEGMENTS:
            raise ValueError(f"Unsupported segment: {segment}")
        if segment in self._subtype_cache:
            return self._subtype_cache[segment]
        if self.subtype_model_dir:
            bundle = self._load_path(self.subtype_model_dir / segment / "model.joblib")
        else:
            resource = files("deepflupred") / f"resources/models/subtype_models/{segment}/model.joblib"
            bundle = self._load_resource(resource)
        self._subtype_cache[segment] = bundle
        return bundle

    def load_pathogenicity(self, host: str) -> dict:
        host = host.lower()
        if host not in PATHOGENICITY_HOSTS:
            raise ValueError(f"Unsupported host: {host}. Choose from {PATHOGENICITY_HOSTS}")
        if host in self._pathogenicity_cache:
            return self._pathogenicity_cache[host]
        if self.pathogenicity_model_dir:
            bundle = self._load_path(self.pathogenicity_model_dir / host / "model.joblib")
        else:
            resource = files("deepflupred") / f"resources/models/pathogenicity_models/{host}/model.joblib"
            bundle = self._load_resource(resource)
        self._pathogenicity_cache[host] = bundle
        return bundle

    def inventory(self) -> list[dict]:
        rows = []
        seg = self.load_segment_model()
        rows.append({
            "model": "segment_model",
            "task": "HA vs NA segment identification",
            "classes": seg.get("classes"),
            "held_out_accuracy": seg.get("held_out_accuracy"),
            "held_out_mcc": seg.get("held_out_mcc"),
        })
        for segment in SEGMENTS:
            bundle = self.load_subtype(segment)
            rows.append({
                "model": f"subtype_models/{segment}",
                "task": f"{segment} subtype classification",
                "classes": bundle.get("classes"),
                "held_out_test_metrics": bundle.get("held_out_test_metrics"),
            })
        for host in PATHOGENICITY_HOSTS:
            bundle = self.load_pathogenicity(host)
            rows.append({
                "model": f"pathogenicity_models/{host}",
                "task": f"HPAI/LPAI classification ({host} cleavage-site window)",
                "classes": ["LPAI", "HPAI"],
                "held_out_test_metrics": bundle.get("held_out_test_metrics"),
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
