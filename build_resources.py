#!/usr/bin/env python3
"""
One-time build script: trains the HA-vs-NA segment BiLSTM (reusing already
computed DNABERT embeddings, no re-encoding) and copies the already-trained
HA/NA subtype and chicken/duck pathogenicity DNABERT+BiLSTM checkpoints from
gisaid_data/new_analysis/ into self-contained bundles under
src/deepflupred/resources/models/.

Not part of the installed package -- run once (or whenever the upstream
gisaid_data/new_analysis models are retrained) to (re)populate
resources/models/. Requires the `tf_metal` conda env (torch>=2.5,
transformers==4.29.2) -- see README.md.
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
# 1. Segment classifier: HA vs NA, trained fresh on cached DNABERT embeddings
#    for the HA@99 / NA@99 representative sequences (no re-encoding needed).
# --------------------------------------------------------------------------
def build_segment_classifier(device):
    print("=== Building segment classifier (HA vs NA) from cached embeddings ===")
    ha_cache = np.load(
        os.path.join(NEW_ANALYSIS, "HA_gene/cdhit_dedup_99pct_h1_h16/dnabert_bilstm_pooled_embeddings_cache.npz"),
        allow_pickle=True,
    )
    na_cache = np.load(
        os.path.join(NEW_ANALYSIS, "NA_gene/cdhit_dedup_99pct_n1_n9/dnabert_bilstm_pooled_embeddings_cache.npz"),
        allow_pickle=True,
    )
    X = np.concatenate([ha_cache["embeddings"], na_cache["embeddings"]], axis=0)
    y = np.array(["HA"] * len(ha_cache["embeddings"]) + ["NA"] * len(na_cache["embeddings"]))
    print(f"  HA embeddings: {len(ha_cache['embeddings'])}   NA embeddings: {len(na_cache['embeddings'])}")

    classes = sorted(set(y))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_to_idx[c] for c in y])

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(y_idx))
    n_test = int(0.1 * len(y_idx))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    model = train_bilstm(X[train_idx], y_idx[train_idx], len(classes), device)
    acc = eval_accuracy(model, X[test_idx], y_idx[test_idx], device)
    print(f"  held-out test: accuracy={acc:.4f} (n={n_test})")

    final_model = train_bilstm(X, y_idx, len(classes), device)
    save_bundle(
        final_model, classes, os.path.join(RESOURCES, "segment_model"),
        metrics={"test_accuracy": float(acc)},
    )
    return {"held_out_accuracy": float(acc), "classes": classes}


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


def main():
    device = get_device()
    print(f"device={device}\n")

    segment_info = build_segment_classifier(device)

    ha_classes = ['H1', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16',
                  'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9']
    na_classes = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']
    copy_subtype_bundle("HA", "HA_gene/cdhit_dedup_99pct_h1_h16", ha_classes)
    copy_subtype_bundle("NA", "NA_gene/cdhit_dedup_99pct_n1_n9", na_classes)

    for host in ("chicken", "duck"):
        copy_pathogenicity_bundle(host)

    import hashlib
    manifest = {"sha256": {}, "segment_model_held_out_accuracy": segment_info["held_out_accuracy"]}
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
