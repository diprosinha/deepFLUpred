#!/usr/bin/env python3
"""
One-time build script: copies the already-trained DNABERT+BiLSTM checkpoints
from gisaid_data/new_analysis/ into self-contained bundles under
src/deepflupred/resources/models/ --

  oneclass/      phase 1, the avian influenza A one-class screen
                 (new_analysis/phase12/train_phase1.py)
  gene_model/    phase 2, the eight-class gene classifier
                 (new_analysis/phase12/train_phase2.py)
  subtype_models/ phase 3, HA (H1-H16) and NA (N1-N9)
  pathogenicity_models/ phase 4, chicken and duck HPAI/LPAI

Every model is trained by its own script upstream; this only assembles the
bundles and writes the SHA-256 manifest. Not part of the installed package --
run once, or whenever an upstream model is retrained. Requires the `tf_metal`
conda env (torch>=2.5, transformers==4.29.2) -- see README.md.

Pre-0.2.0 resources (segment_model/, oneclass/stage0_*, oneclass/stage1_*) are
removed if present: the eight-class gene model subsumes the HA-vs-NA segment
classifier, and the single phase-1 screen replaces the two old gates.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from deepflupred.dnabert_bilstm import BiLSTMOnEmbeddings, get_device  # noqa: E402

GISAID_ROOT = os.path.dirname(HERE)  # deepFLUpred lives inside gisaid_data/
NEW_ANALYSIS = os.path.join(GISAID_ROOT, "new_analysis")
RESOURCES = os.path.join(HERE, "src", "deepflupred", "resources", "models")
SEED = 42
LSTM_HIDDEN = 64
BATCH_SIZE = 128
LR = 1e-3
TARGET_GRAD_STEPS = 15000
MIN_EPOCHS, MAX_EPOCHS = 15, 100


def epochs_for(n_train, batch_size=BATCH_SIZE):
    steps_per_epoch = max(1, -(-n_train // batch_size))
    return int(min(MAX_EPOCHS, max(MIN_EPOCHS, round(TARGET_GRAD_STEPS / steps_per_epoch))))


def train_bilstm(X_tr, y_tr_idx, n_classes, device):
    epochs = epochs_for(len(y_tr_idx))
    model = BiLSTMOnEmbeddings(X_tr.shape[-1], LSTM_HIDDEN, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    class_counts = np.bincount(y_tr_idx, minlength=n_classes)
    weights = torch.tensor(len(y_tr_idx) / np.maximum(class_counts, 1), dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    X = torch.tensor(X_tr, dtype=torch.float32)
    y = torch.tensor(y_tr_idx, dtype=torch.long)
    n = X.shape[0]
    print(f"    training for {epochs} epochs on {n} sequences...")
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X[idx].to(device), y[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def eval_accuracy(model, X, y_idx, device, batch_size=BATCH_SIZE):
    Xt = torch.tensor(X, dtype=torch.float32)
    correct = 0
    for i in range(0, Xt.shape[0], batch_size):
        xb = Xt[i:i + batch_size].to(device)
        preds = model(xb).argmax(1).cpu().numpy()
        correct += (preds == y_idx[i:i + batch_size]).sum()
    return correct / len(y_idx)


def save_bundle(model, classes, out_dir, metrics=None):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "classes.json"), "w") as fh:
        json.dump(list(classes), fh)
    if metrics is not None:
        with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
    print(f"  wrote {out_dir}/model.pt + classes.json" + ("+ metrics.json" if metrics else ""))


def dnabert_bilstm_test_metrics(summary_csv_path):
    """Pull the DNABERT_BiLSTM row's held-out test metrics from an existing
    new_analysis model_comparison_summary.csv."""
    df = pd.read_csv(summary_csv_path)
    row = df[df["model"] == "DNABERT_BiLSTM"].iloc[0]
    return {
        "test_accuracy": float(row["test_accuracy"]),
        "test_f1_macro": float(row["test_f1_macro"]),
        "test_mcc": float(row["test_mcc"]),
    }


# --------------------------------------------------------------------------
# 1. Phase 2 gene classifier: copy the eight-class bundle trained by
#    ../new_analysis/phase12/train_phase2.py on the cluster-disjoint splits.
#    That script owns the training; this only copies the finished bundle.
# --------------------------------------------------------------------------
def copy_gene_bundle():
    print("=== Copying phase-2 eight-class gene classifier ===")
    import shutil

    source_dir = os.path.join(NEW_ANALYSIS, "phase12", "models", "phase2")
    out_dir = os.path.join(RESOURCES, "gene_model")
    os.makedirs(out_dir, exist_ok=True)

    missing = [
        name for name in ("model.pt", "classes.json", "metrics.json")
        if not os.path.exists(os.path.join(source_dir, name))
    ]
    if missing:
        raise SystemExit(
            f"Missing phase-2 artifacts in {os.path.relpath(source_dir, GISAID_ROOT)}: "
            f"{missing}. Run new_analysis/phase12/train_phase2.py first."
        )
    for name in ("model.pt", "classes.json", "metrics.json"):
        shutil.copy2(os.path.join(source_dir, name), os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "metrics.json")) as fh:
        metrics = json.load(fh)
    with open(os.path.join(out_dir, "classes.json")) as fh:
        classes = json.load(fh)
    print(f"  classes ({len(classes)}): {classes}")
    print(f"  test accuracy={metrics.get('test_accuracy'):.4f}  "
          f"macro-F1={metrics.get('test_macro_f1'):.4f}")
    print(f"  wrote {out_dir}/model.pt + classes.json + metrics.json")
    return {"classes": classes, "test_accuracy": metrics.get("test_accuracy")}


# --------------------------------------------------------------------------
# 2. Subtype classifiers: copy the already-trained final DNABERT+BiLSTM
#    checkpoint (fit on the full 90% cluster-disjoint training set) from
#    new_analysis, at 99% CD-HIT identity.
# --------------------------------------------------------------------------
def copy_subtype_bundle(segment, source_dir, classes):
    print(f"=== Copying {segment} subtype DNABERT+BiLSTM checkpoint ===")
    src = os.path.join(NEW_ANALYSIS, source_dir, "DNABERT_BiLSTM", "model.pt")
    sd = torch.load(src, map_location="cpu")
    out_dir = os.path.join(RESOURCES, "subtype_models", segment)
    metrics = dnabert_bilstm_test_metrics(os.path.join(NEW_ANALYSIS, source_dir, "model_comparison_summary.csv"))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(sd, os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "classes.json"), "w") as fh:
        json.dump(list(classes), fh)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  classes ({len(classes)}): {classes}")
    print(f"  test metrics: {metrics}")
    print(f"  wrote {out_dir}/model.pt + classes.json + metrics.json")


# --------------------------------------------------------------------------
# 3. Pathogenicity classifiers: copy chicken/duck DNABERT+BiLSTM checkpoints.
#    No human model exists for this architecture (out of scope by request).
# --------------------------------------------------------------------------
def copy_pathogenicity_bundle(host):
    print(f"=== Copying pathogenicity DNABERT+BiLSTM checkpoint for host={host} ===")
    src = os.path.join(NEW_ANALYSIS, "hpli_lpai", host, "DNABERT_BiLSTM", "model.pt")
    sd = torch.load(src, map_location="cpu")
    out_dir = os.path.join(RESOURCES, "pathogenicity_models", host)
    metrics = dnabert_bilstm_test_metrics(
        os.path.join(NEW_ANALYSIS, "hpli_lpai", host, "model_comparison_summary.csv")
    )
    os.makedirs(out_dir, exist_ok=True)
    torch.save(sd, os.path.join(out_dir, "model.pt"))
    classes = ["HPAI", "LPAI"]
    with open(os.path.join(out_dir, "classes.json"), "w") as fh:
        json.dump(classes, fh)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  test metrics: {metrics}")
    print(f"  wrote {out_dir}/model.pt + classes.json + metrics.json")



# --------------------------------------------------------------------------
# 4. Phase 1 one-class screen, trained by
#    ../new_analysis/phase12/train_phase1.py. That script owns the training;
#    this only copies the finished bundle into the package.
# --------------------------------------------------------------------------
def copy_oneclass_bundle():
    print("=== Copying phase-1 one-class screen ===")
    import shutil

    source_dir = os.path.join(NEW_ANALYSIS, "phase12", "models", "phase1")
    out_dir = os.path.join(RESOURCES, "oneclass")
    os.makedirs(out_dir, exist_ok=True)

    wanted = ["phase1_oneclass_bilstm.pt", "phase1_metadata.json"]
    missing = []
    for filename in wanted:
        source = os.path.join(source_dir, filename)
        if not os.path.exists(source):
            missing.append(os.path.relpath(source, GISAID_ROOT))
            continue
        shutil.copy2(source, os.path.join(out_dir, filename))
        print(f"  copied {filename}")
    if missing:
        print("  WARNING: missing phase-1 artifacts, the screen will be unavailable:")
        for item in missing:
            print(f"    {item}")
    return {"missing": missing}


def remove_superseded_resources():
    """Drop the pre-0.2.0 bundles so a rebuilt package cannot ship both the old
    HA-vs-NA segment model and the eight-class gene model, or both one-class
    gate generations."""
    import shutil

    print("=== Removing superseded pre-0.2.0 resources ===")
    for relative in ("segment_model",):
        path = os.path.join(RESOURCES, relative)
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  removed {relative}/")
    for relative in (
        "oneclass/stage0_oneclass.joblib",
        "oneclass/stage0_metadata.json",
        "oneclass/stage1_oneclass_bilstm.pt",
        "oneclass/stage1_metadata.json",
    ):
        path = os.path.join(RESOURCES, relative)
        if os.path.exists(path):
            os.remove(path)
            print(f"  removed {relative}")


def main():
    device = get_device()
    print(f"device={device}\n")

    remove_superseded_resources()
    gene_info = copy_gene_bundle()

    ha_classes = ['H1', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16',
                  'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9']
    na_classes = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']
    copy_subtype_bundle("HA", "HA_gene/cdhit_dedup_99pct_h1_h16", ha_classes)
    copy_subtype_bundle("NA", "NA_gene/cdhit_dedup_99pct_n1_n9", na_classes)

    for host in ("chicken", "duck"):
        copy_pathogenicity_bundle(host)

    copy_oneclass_bundle()

    import hashlib
    manifest = {"sha256": {}, "gene_model_test_accuracy": gene_info["test_accuracy"]}
    for root, _, files in os.walk(RESOURCES):
        for fname in files:
            if fname in ("manifest.json",):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, RESOURCES)
            manifest["sha256"][rel] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest_path = os.path.join(RESOURCES, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
