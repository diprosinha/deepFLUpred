"""FluMAPPER prediction pipeline.

Three stages, run in order, each gating the next:

  1. identify_segment   -- is this sequence HA, NA, or neither?
  2. predict_subtype    -- if HA/NA: which subtype (H1-H16 / N1-N9)?
  3. predict_pathogenicity -- if HA: HPAI or LPAI, from the HA1/HA2
     cleavage site (both a biological rule -- polybasic/monobasic motif --
     and a per-host machine-learning model, reported together)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from flumapper.cleavage_site import analyze_sequence
from flumapper.constants import (
    HA_LENGTH_RANGE,
    NA_LENGTH_RANGE,
    PATHOGENICITY_HOSTS,
    PATHOGENICITY_LABEL_MAP,
)
from flumapper.features import extract_features, kmer_only_features
from flumapper.model_store import ModelStore

SEGMENT_CONFIDENCE_THRESHOLD = 0.75
SUBTYPE_CONFIDENCE_THRESHOLD = 0.50


def parse_fasta(path) -> list[tuple[str, str]]:
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


def _sequence_id(header: str) -> str:
    """First pipe-delimited field, or the whole header if there is none."""
    return header.split("|", 1)[0].strip() or header


def identify_segment(seq: str, store: ModelStore) -> dict:
    bundle = store.load_segment_model()
    model = bundle["model"]
    classes = list(model.classes_)

    X = kmer_only_features(seq).reshape(1, -1)
    proba = model.predict_proba(X)[0]
    top_idx = int(np.argmax(proba))
    top_class, confidence = classes[top_idx], float(proba[top_idx])

    seq_len = len(seq.strip())
    length_ok = (
        (HA_LENGTH_RANGE[0] <= seq_len <= HA_LENGTH_RANGE[1])
        if top_class == "HA"
        else (NA_LENGTH_RANGE[0] <= seq_len <= NA_LENGTH_RANGE[1])
    )

    resolved = confidence >= SEGMENT_CONFIDENCE_THRESHOLD and length_ok
    return {
        "segment": top_class if resolved else None,
        "segment_confidence": confidence,
        "segment_length_plausible": length_ok,
        "segment_probabilities": dict(zip(classes, (float(p) for p in proba))),
    }


def predict_subtype(seq: str, segment: str, store: ModelStore) -> dict:
    bundle = store.load_subtype(segment)
    model = bundle["model"]
    target_len = bundle["target_len"]
    mask = bundle["selected_features_mask"]
    classes = list(model.classes_)

    X = extract_features(seq, target_len)[mask].reshape(1, -1)
    proba = model.predict_proba(X)[0]
    top_idx = int(np.argmax(proba))
    top_class, confidence = classes[top_idx], float(proba[top_idx])

    return {
        "subtype": top_class,
        "subtype_confidence": confidence,
        "subtype_resolved": confidence >= SUBTYPE_CONFIDENCE_THRESHOLD,
        "subtype_probabilities": dict(zip(classes, (float(p) for p in proba))),
    }


def predict_pathogenicity(seq: str, store: ModelStore, hosts: list[str] | None = None) -> dict:
    """Runs on a full-length HA nucleotide sequence: locates the HA1/HA2
    cleavage site, reports the rule-based Polybasic/Monobasic call, and
    (when the 110nt ML window is fully covered by the input) the
    per-host RandomForest HPAI/LPAI prediction."""
    hosts = hosts or list(PATHOGENICITY_HOSTS)
    site = analyze_sequence(seq)

    result = {
        "cleavage_site_resolved": site is not None,
        "cleavage_site_classification": None,
        "cleavage_site_motif": None,
        "cleavage_site_window": None,
        "pathogenicity_ml": {},
        "pathogenicity_consensus": None,
    }
    if site is None:
        return result

    result["cleavage_site_classification"] = site["classification"]
    result["cleavage_site_motif"] = site["motif_aa"]
    result["cleavage_site_window"] = site["window_seq"]

    window = site["window_seq"]
    if window is None:
        return result

    calls = []
    for host in hosts:
        bundle = store.load_pathogenicity(host)
        model = bundle["model"]
        target_len = bundle["target_len"]
        mask = bundle["selected_features_mask"]
        X = extract_features(window, target_len)[mask].reshape(1, -1)
        proba = model.predict_proba(X)[0]
        model_classes = list(model.classes_)  # integer labels, see PATHOGENICITY_LABEL_MAP
        top_idx = int(np.argmax(proba))
        top_label = PATHOGENICITY_LABEL_MAP[model_classes[top_idx]]
        confidence = float(proba[top_idx])
        result["pathogenicity_ml"][host] = {
            "prediction": top_label,
            "confidence": confidence,
            "probabilities": {
                PATHOGENICITY_LABEL_MAP[c]: float(p) for c, p in zip(model_classes, proba)
            },
        }
        calls.append(top_label)

    if calls:
        result["pathogenicity_consensus"] = max(set(calls), key=calls.count)
        result["pathogenicity_ml_agreement"] = calls.count(result["pathogenicity_consensus"]) / len(calls)
    return result


def predict_sequence(
    seq: str,
    identifier: str,
    store: ModelStore,
    expected_segment: str | None = None,
    pathogenicity_hosts: list[str] | None = None,
) -> dict:
    seq = seq.strip()
    row: dict = {"sequence_id": identifier, "sequence_length": len(seq)}

    seg = identify_segment(seq, store)
    row["segment"] = seg["segment"]
    row["segment_confidence"] = round(seg["segment_confidence"], 4)

    if seg["segment"] is None:
        row["status"] = "segment_unresolved"
        row["explanation"] = (
            "Not confidently identified as HA or NA "
            f"(top-class confidence={seg['segment_confidence']:.3f}, "
            f"length={row['sequence_length']}bp)."
        )
        return row

    if expected_segment and seg["segment"] != expected_segment.upper():
        row["status"] = "expected_segment_mismatch"
        row["explanation"] = (
            f"Expected {expected_segment.upper()} but classified as {seg['segment']} "
            f"(confidence={seg['segment_confidence']:.3f})."
        )

    subtype = predict_subtype(seq, seg["segment"], store)
    row["subtype"] = subtype["subtype"]
    row["subtype_confidence"] = round(subtype["subtype_confidence"], 4)

    if seg["segment"] == "HA":
        path = predict_pathogenicity(seq, store, hosts=pathogenicity_hosts)
        row["cleavage_site_resolved"] = path["cleavage_site_resolved"]
        row["cleavage_site_classification"] = path["cleavage_site_classification"]
        row["cleavage_site_motif"] = path["cleavage_site_motif"]
        row["pathogenicity_consensus"] = path["pathogenicity_consensus"]
        for host, call in path["pathogenicity_ml"].items():
            row[f"pathogenicity_{host}"] = call["prediction"]
            row[f"pathogenicity_{host}_confidence"] = round(call["confidence"], 4)

    if "status" not in row:
        row["status"] = "subtype_assigned"
        explanation = (
            f"Identified as {seg['segment']} (confidence={seg['segment_confidence']:.3f}); "
            f"subtype {subtype['subtype']} (confidence={subtype['subtype_confidence']:.3f})."
        )
        if seg["segment"] == "HA":
            if row.get("cleavage_site_resolved"):
                explanation += (
                    f" Cleavage site: {row['cleavage_site_classification']}"
                    f" (motif={row['cleavage_site_motif']})."
                )
                if row.get("pathogenicity_consensus"):
                    explanation += f" Pathogenicity (ML consensus): {row['pathogenicity_consensus']}."
            else:
                explanation += " Cleavage site could not be located; pathogenicity not assessed."
        row["explanation"] = explanation

    return row


def predict_sequences(
    sequences: list[str],
    identifiers: list[str] | None = None,
    expected_segment: str | None = None,
    pathogenicity_hosts: list[str] | None = None,
    segment_model: str | Path | None = None,
    subtype_model_dir: str | Path | None = None,
    pathogenicity_model_dir: str | Path | None = None,
) -> pd.DataFrame:
    store = ModelStore(
        segment_model=segment_model,
        subtype_model_dir=subtype_model_dir,
        pathogenicity_model_dir=pathogenicity_model_dir,
    )
    identifiers = identifiers or [f"seq{i+1}" for i in range(len(sequences))]
    rows = [
        predict_sequence(seq, ident, store, expected_segment, pathogenicity_hosts)
        for ident, seq in zip(identifiers, sequences)
    ]
    return pd.DataFrame(rows)


def predict_fasta(
    fasta_path,
    output=None,
    expected_segment: str | None = None,
    pathogenicity_hosts: list[str] | None = None,
    segment_model: str | Path | None = None,
    subtype_model_dir: str | Path | None = None,
    pathogenicity_model_dir: str | Path | None = None,
) -> pd.DataFrame:
    records = parse_fasta(fasta_path)
    headers = [h for h, _ in records]
    seqs = [s for _, s in records]

    raw_ids = [_sequence_id(h) for h in headers]
    ids = raw_ids if len(set(raw_ids)) == len(raw_ids) else headers

    table = predict_sequences(
        seqs,
        identifiers=ids,
        expected_segment=expected_segment,
        pathogenicity_hosts=pathogenicity_hosts,
        segment_model=segment_model,
        subtype_model_dir=subtype_model_dir,
        pathogenicity_model_dir=pathogenicity_model_dir,
    )
    if output:
        os.makedirs(os.path.dirname(str(output)) or ".", exist_ok=True)
        table.to_csv(output, index=False)
    return table
