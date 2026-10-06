"""deepFLUpred prediction pipeline.

Four phases, run in order, each gating the next:

  1. AvianInfluenzaScreen.screen -- one-class: does this belong to any gene of
     avian influenza A? Trained on all eight segments, positives only, so a
     plant gene or a coronavirus is turned away before any classifier is
     consulted rather than being forced into a gene call. See oneclass.py.
  2. identify_gene       -- which gene is it? An eight-class choice across
     PB2/PB1/PA/HA/NP/NA/MP/NS. For the six internal segments this is the
     final answer: influenza subtyping is defined only for HA and NA.
  3. predict_subtype     -- if HA or NA: which subtype (H1-H16 / N1-N9)?
  4. predict_pathogenicity -- if HA: HPAI or LPAI, from the HA1/HA2 cleavage
     site (both a biological rule -- polybasic/monobasic motif -- and a
     per-host DNABERT+BiLSTM model, reported together).

A phase-1 rejection says only that the sequence is unlike the avian influenza A
sequences the screen was trained on; it cannot identify what the sequence
actually is. Phase 1 also screens for influenza A genes rather than for host, so
passing means "influenza A-like", not "confirmed avian origin".

Every phase runs on DNABERT-2 embeddings (frozen encoder, adaptive-pooled to a
fixed 16-step sequence) classified by a trained BiLSTM head -- see
dnabert_bilstm.py. Phases 1, 2 and 3 all consume an identical encoding of the
full input sequence, so that embedding is computed once per sequence and reused
across all three rather than re-running the transformer three times. Phase 4
encodes only the extracted 110nt cleavage-site window.

Prediction is alignment-free: no BLAST, no pairwise identity.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from deepflupred.cleavage_site import analyze_sequence
from deepflupred.constants import (
    GENE_LENGTH_RANGES,
    PATHOGENICITY_HOSTS,
    SUBTYPEABLE_GENES,
)
from deepflupred.dnabert_bilstm import pooled_embed_sequences, predict_proba
from deepflupred.model_store import ModelStore

GENE_CONFIDENCE_THRESHOLD = 0.75
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


def embed_one(seq: str, store: ModelStore) -> np.ndarray:
    """DNABERT tokenizer -> DNABERT Transformer -> hidden states -> adaptive
    pooling -> a single (16, 768) numerical feature-vector sequence."""
    tok, model = store.encoder
    return pooled_embed_sequences([seq], tok, model, store.device)[0]


def identify_gene(embedding: np.ndarray, seq_len: int, store: ModelStore) -> dict:
    """Phase 2: which of the eight influenza A genes is this?

    A call is reported only when the top-class probability clears
    GENE_CONFIDENCE_THRESHOLD. Length is recorded as a plausibility flag
    against that gene's window but does not veto the call: the windows overlap
    between genes, and a partial record is still worth a reported gene with a
    flag rather than a refusal.
    """
    bundle = store.load_gene_model()
    classes = bundle["classes"]

    proba = predict_proba(bundle["model"], embedding, store.device)
    order = np.argsort(proba)[::-1]
    top_idx = int(order[0])
    top_class, confidence = classes[top_idx], float(proba[top_idx])
    runner_up = classes[int(order[1])] if len(order) > 1 else None

    window = GENE_LENGTH_RANGES.get(top_class)
    length_ok = bool(window and window[0] <= seq_len <= window[1])

    return {
        "gene": top_class if confidence >= GENE_CONFIDENCE_THRESHOLD else None,
        "gene_top_class": top_class,
        "gene_confidence": confidence,
        "gene_runner_up": runner_up,
        "gene_length_plausible": length_ok,
        "gene_probabilities": dict(zip(classes, (float(p) for p in proba))),
    }


def predict_subtype(embedding: np.ndarray, gene: str, store: ModelStore) -> dict:
    """Phase 3: HA -> H1-H16, NA -> N1-N9."""
    bundle = store.load_subtype(gene)
    classes = bundle["classes"]

    proba = predict_proba(bundle["model"], embedding, store.device)
    top_idx = int(np.argmax(proba))
    top_class, confidence = classes[top_idx], float(proba[top_idx])

    return {
        "subtype": top_class,
        "subtype_confidence": confidence,
        "subtype_resolved": confidence >= SUBTYPE_CONFIDENCE_THRESHOLD,
        "subtype_probabilities": dict(zip(classes, (float(p) for p in proba))),
    }


def predict_pathogenicity(seq: str, store: ModelStore, hosts: list[str] | None = None) -> dict:
    """Phase 4, on a full-length HA nucleotide sequence: locates the HA1/HA2
    cleavage site, reports the rule-based Polybasic/Monobasic call, and
    (when the 110nt ML window is fully covered by the input) the per-host
    DNABERT+BiLSTM HPAI/LPAI prediction."""
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

    window_embedding = embed_one(window, store)

    calls = []
    for host in hosts:
        bundle = store.load_pathogenicity(host)
        classes = bundle["classes"]
        proba = predict_proba(bundle["model"], window_embedding, store.device)
        top_idx = int(np.argmax(proba))
        top_label = classes[top_idx]
        confidence = float(proba[top_idx])
        result["pathogenicity_ml"][host] = {
            "prediction": top_label,
            "confidence": confidence,
            "probabilities": dict(zip(classes, (float(p) for p in proba))),
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
    expected_gene: str | None = None,
    pathogenicity_hosts: list[str] | None = None,
    run_screen: bool = True,
) -> dict:
    seq = seq.strip()
    row: dict = {"sequence_id": identifier, "sequence_length": len(seq)}

    embedding = embed_one(seq, store)

    # -- phase 1: is this any gene of avian influenza A? ----------------------
    if run_screen:
        verdict = store.screen.screen(embedding)
        if verdict is not None:
            row.update(verdict.as_row("avian_influenza"))
            if not verdict.accepted:
                row["status"] = "not_avian_influenza"
                row["explanation"] = (
                    "Rejected by the phase-1 avian influenza A one-class screen "
                    f"(score={verdict.score:.4f} below threshold {verdict.threshold:.4f}). "
                    "The sequence is unlike any of the eight avian influenza A genes the "
                    "screen was trained on; no gene or subtype was assigned."
                )
                return row

    # -- phase 2: which gene? -------------------------------------------------
    gene_call = identify_gene(embedding, len(seq), store)
    row["gene"] = gene_call["gene"]
    row["gene_confidence"] = round(gene_call["gene_confidence"], 4)
    row["gene_length_plausible"] = gene_call["gene_length_plausible"]

    if gene_call["gene"] is None:
        row["status"] = "gene_unresolved"
        row["explanation"] = (
            "Passed the phase-1 screen but no gene was identified with sufficient confidence "
            f"(best guess {gene_call['gene_top_class']} at "
            f"{gene_call['gene_confidence']:.3f}, below the "
            f"{GENE_CONFIDENCE_THRESHOLD:.2f} floor; runner-up "
            f"{gene_call['gene_runner_up']}, length={row['sequence_length']}bp)."
        )
        return row

    gene = gene_call["gene"]
    if expected_gene and gene != expected_gene.upper():
        row["status"] = "expected_gene_mismatch"
        row["explanation"] = (
            f"Expected {expected_gene.upper()} but classified as {gene} "
            f"(confidence={gene_call['gene_confidence']:.3f})."
        )

    # The six internal segments are a complete answer: subtyping is defined
    # only for HA and NA, so the pipeline stops here rather than inventing one.
    if gene not in SUBTYPEABLE_GENES:
        if "status" not in row:
            row["status"] = "gene_identified"
            row["explanation"] = (
                f"Identified as {gene} (confidence={gene_call['gene_confidence']:.3f}). "
                "Subtype and pathogenicity are defined only for HA and NA, so no further "
                "phase was run."
            )
        return row

    # -- phase 3: subtype -----------------------------------------------------
    subtype = predict_subtype(embedding, gene, store)
    row["subtype"] = subtype["subtype"]
    row["subtype_confidence"] = round(subtype["subtype_confidence"], 4)

    # -- phase 4: pathogenicity (HA only) -------------------------------------
    if gene == "HA":
        path = predict_pathogenicity(seq, store, hosts=pathogenicity_hosts)
        row["cleavage_site_resolved"] = path["cleavage_site_resolved"]
        row["cleavage_site_classification"] = path["cleavage_site_classification"]
        row["cleavage_site_motif"] = path["cleavage_site_motif"]
        row["pathogenicity_consensus"] = path["pathogenicity_consensus"]

    if "status" not in row:
        row["status"] = "subtype_assigned"
        explanation = (
            f"Identified as {gene} (confidence={gene_call['gene_confidence']:.3f}); "
            f"subtype {subtype['subtype']} (confidence={subtype['subtype_confidence']:.3f})."
        )
        if gene == "HA":
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
    expected_gene: str | None = None,
    pathogenicity_hosts: list[str] | None = None,
    gene_model: str | Path | None = None,
    subtype_model_dir: str | Path | None = None,
    pathogenicity_model_dir: str | Path | None = None,
    oneclass_model_dir: str | Path | None = None,
    run_screen: bool = True,
) -> pd.DataFrame:
    store = ModelStore(
        gene_model=gene_model,
        subtype_model_dir=subtype_model_dir,
        pathogenicity_model_dir=pathogenicity_model_dir,
        oneclass_model_dir=oneclass_model_dir,
    )
    identifiers = identifiers or [f"seq{i+1}" for i in range(len(sequences))]
    rows = [
        predict_sequence(seq, ident, store, expected_gene, pathogenicity_hosts, run_screen)
        for ident, seq in zip(identifiers, sequences)
    ]
    return pd.DataFrame(rows)


def predict_fasta(
    fasta_path,
    output=None,
    expected_gene: str | None = None,
    pathogenicity_hosts: list[str] | None = None,
    gene_model: str | Path | None = None,
    subtype_model_dir: str | Path | None = None,
    pathogenicity_model_dir: str | Path | None = None,
    oneclass_model_dir: str | Path | None = None,
    run_screen: bool = True,
) -> pd.DataFrame:
    records = parse_fasta(fasta_path)
    headers = [h for h, _ in records]
    seqs = [s for _, s in records]

    raw_ids = [_sequence_id(h) for h in headers]
    ids = raw_ids if len(set(raw_ids)) == len(raw_ids) else headers

    table = predict_sequences(
        seqs,
        identifiers=ids,
        expected_gene=expected_gene,
        pathogenicity_hosts=pathogenicity_hosts,
        gene_model=gene_model,
        subtype_model_dir=subtype_model_dir,
        pathogenicity_model_dir=pathogenicity_model_dir,
        oneclass_model_dir=oneclass_model_dir,
        run_screen=run_screen,
    )
    if output:
        os.makedirs(os.path.dirname(str(output)) or ".", exist_ok=True)
        table.to_csv(output, index=False)
    return table
