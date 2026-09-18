#!/usr/bin/env python3
"""
One-time build script: trains the HA-vs-NA segment classifier and bundles it,
plus the already-trained subtype and pathogenicity RandomForest models, into
self-contained joblib bundles under src/flumapper/resources/models/.

Not part of the installed package -- run once (or whenever the upstream
gisaid_data models are retrained) to (re)populate resources/models/.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from flumapper.features import extract_features, feature_names, kmer_only_features  # noqa: E402

GISAID_ROOT = os.path.dirname(HERE)  # FluMAPPER now lives inside gisaid_data/
RESOURCES = os.path.join(HERE, "src", "flumapper", "resources", "models")
SEED = 42


def parse_fasta(path):
    records = []
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:]
                chunks = []
            elif line:
                chunks.append(line)
        if header is not None:
            records.append((header, "".join(chunks)))
    return records


def selection_mask_from_csv(csv_path, target_len):
    """Reconstruct the boolean feature-selection mask, in canonical feature
    order, from a saved feature_importances.csv (which is sorted by
    importance, not canonical order -- so we join by feature name)."""
    df = pd.read_csv(csv_path)
    selected_by_name = dict(zip(df["feature"], df["selected"]))
    names = feature_names(target_len)
    return np.array([bool(selected_by_name[n]) for n in names], dtype=bool)


# --------------------------------------------------------------------------
# 1. Segment classifier: HA vs NA, length-invariant dinuc+trinuc features
# --------------------------------------------------------------------------
def build_segment_classifier():
    print("=== Building segment classifier (HA vs NA) ===")
    ha_fasta = os.path.join(
        GISAID_ROOT, "HA_gene", "multiclass_subtype", "cdhit_dedup_99pct_h1_h16", "multiclass_train.fasta"
    )
    na_fasta = os.path.join(
        GISAID_ROOT, "NA_gene", "multiclass_subtype", "cdhit_dedup_99pct_n1_n9", "multiclass_train.fasta"
    )
    ha_recs = parse_fasta(ha_fasta)
    na_recs = parse_fasta(na_fasta)
    print(f"  HA representatives: {len(ha_recs)}   NA representatives: {len(na_recs)}")

    seqs = [s for _, s in ha_recs] + [s for _, s in na_recs]
    labels = ["HA"] * len(ha_recs) + ["NA"] * len(na_recs)

    X = np.vstack([kmer_only_features(s) for s in seqs])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    print(f"  held-out test: accuracy={acc:.4f} mcc={mcc:.4f} (n={len(y_test)})")

    clf_full = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1)
    clf_full.fit(X, y)

    bundle = {
        "model": clf_full,
        "model_family": "RandomForestClassifier",
        "feature_type": "kmer_only",  # dinuc+trinuc, length-invariant
        "classes": sorted(set(labels)),
        "held_out_accuracy": float(acc),
        "held_out_mcc": float(mcc),
        "n_train": len(seqs),
    }
    out_path = os.path.join(RESOURCES, "segment_model", "model.joblib")
    joblib.dump(bundle, out_path)
    print(f"  wrote {out_path}")
    return {"segment_model/model.joblib": out_path}


# --------------------------------------------------------------------------
# 2. Subtype classifiers (HA: H1-H16, NA: N1-N9) -- reuse already-trained
#    99%-CD-HIT-dedup RandomForest models, bundled with their selection mask
# --------------------------------------------------------------------------
def build_subtype_bundle(segment, run_dir, target_len):
    print(f"=== Bundling {segment} subtype classifier (from {run_dir}) ===")
    model_path = os.path.join(run_dir, "ml_results", "RandomForest", "model.joblib")
    fi_path = os.path.join(run_dir, "ml_results", "feature_importances.csv")
    test_metrics_path = os.path.join(run_dir, "ml_results", "RandomForest", "test_metrics.csv")

    model = joblib.load(model_path)
    mask = selection_mask_from_csv(fi_path, target_len)
    test_metrics = pd.read_csv(test_metrics_path, index_col=0).iloc[:, 0].to_dict()

    bundle = {
        "model": model,
        "model_family": "RandomForestClassifier",
        "feature_type": "chem_dinuc_trinuc",
        "target_len": target_len,
        "selected_features_mask": mask,
        "classes": sorted(model.classes_.tolist()),
        "source_run_dir": os.path.relpath(run_dir, GISAID_ROOT),
        "held_out_test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    out_dir = os.path.join(RESOURCES, "subtype_models", segment)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(bundle, out_path)
    print(f"  classes: {bundle['classes']}")
    print(f"  held-out test: {bundle['held_out_test_metrics']}")
    print(f"  wrote {out_path}")
    return {f"subtype_models/{segment}/model.joblib": out_path}


# --------------------------------------------------------------------------
# 3. Pathogenicity classifiers (HPAI vs LPAI), per host -- reuse
#    already-trained cleavage-site-window RandomForest models
# --------------------------------------------------------------------------
def build_pathogenicity_bundle(host):
    print(f"=== Bundling pathogenicity classifier for host={host} ===")
    host_dir = os.path.join(GISAID_ROOT, "hpli_lpai", host)
    model_path = os.path.join(host_dir, "ml_results", "RandomForest_model.joblib")
    fi_path = os.path.join(host_dir, "ml_results", "feature_importances.csv")
    test_metrics_path = os.path.join(host_dir, "ml_results", "RandomForest_test_metrics.csv")

    model = joblib.load(model_path)
    mask = selection_mask_from_csv(fi_path, 110)
    test_metrics = pd.read_csv(test_metrics_path, index_col=0).iloc[:, 0].to_dict()

    bundle = {
        "model": model,
        "model_family": "RandomForestClassifier",
        "feature_type": "chem_dinuc_trinuc",
        "target_len": 110,
        "selected_features_mask": mask,
        "classes": sorted(model.classes_.tolist()),
        "host": host,
        "source_run_dir": os.path.relpath(host_dir, GISAID_ROOT),
        "held_out_test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    out_dir = os.path.join(RESOURCES, "pathogenicity_models", host)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(bundle, out_path)
    print(f"  classes: {bundle['classes']}")
    print(f"  held-out test: {bundle['held_out_test_metrics']}")
    print(f"  wrote {out_path}")
    return {f"pathogenicity_models/{host}/model.joblib": out_path}


def main():
    manifest_paths = {}
    manifest_paths.update(build_segment_classifier())
    manifest_paths.update(build_subtype_bundle(
        "HA",
        os.path.join(GISAID_ROOT, "HA_gene", "multiclass_subtype", "cdhit_dedup_99pct_h1_h16"),
        1800,
    ))
    manifest_paths.update(build_subtype_bundle(
        "NA",
        os.path.join(GISAID_ROOT, "NA_gene", "multiclass_subtype", "cdhit_dedup_99pct_n1_n9"),
        1500,
    ))
    for host in ("chicken", "duck", "human"):
        manifest_paths.update(build_pathogenicity_bundle(host))

    import hashlib
    manifest = {"sha256": {}}
    for rel_path, abs_path in manifest_paths.items():
        manifest["sha256"][rel_path] = hashlib.sha256(open(abs_path, "rb").read()).hexdigest()
    manifest_path = os.path.join(RESOURCES, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
