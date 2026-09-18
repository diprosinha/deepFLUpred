"""Alignment-free sequence features used by every deepFLUpred model.

Exactly reproduces the feature extraction used to train the bundled models
(see gisaid_data/HA_gene/multiclass_subtype/run_h1_h16_cdhit_dedup.py,
NA_gene/multiclass_subtype/run_n1_n9_cdhit_dedup.py, and
gisaid_data/hpli_lpai/*/train.py) so predictions on new sequences are
computed identically to how each model was trained and evaluated.
"""
from __future__ import annotations

import numpy as np

NUCS = "ACGT"
DINUCS = [a + b for a in NUCS for b in NUCS]
TRINUCS = [a + b + c for a in NUCS for b in NUCS for c in NUCS]

# (ring structure, functional group, H-bond count)
CHEM_PROP = {
    "A": (1, 1, 0),
    "C": (0, 1, 1),
    "G": (1, 0, 1),
    "T": (0, 0, 0),
}


def clean_sequence(seq: str) -> str:
    return str(seq).upper().replace("U", "T")


def fix_length(seq: str, target_len: int) -> str:
    """Pad with 'N' or truncate to a fixed length, matching training-time preprocessing."""
    seq = clean_sequence(seq)
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + "N" * (target_len - len(seq))


def chemical_property_features(seq: str) -> list[float]:
    feats: list[float] = []
    for base in seq:
        feats.extend(CHEM_PROP.get(base, (0, 0, 0)))
    return feats


def kmer_frequency(seq: str, k: int, kmers: list[str]) -> list[float]:
    counts = {kmer: 0 for kmer in kmers}
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if kmer in counts:
            counts[kmer] += 1
            total += 1
    if total == 0:
        return [0.0] * len(kmers)
    return [counts[kmer] / total for kmer in kmers]


def kmer_only_features(seq: str) -> np.ndarray:
    """Length-invariant dinucleotide + trinucleotide frequency vector (80-dim).

    Used by the segment-identification model: unlike the chemical-property
    features below, it needs no fixed padded length, so it can be computed
    identically for HA (~1500-1800bp) and NA (~1300-1500bp) candidates."""
    seq = clean_sequence(seq)
    di = kmer_frequency(seq, 2, DINUCS)
    tri = kmer_frequency(seq, 3, TRINUCS)
    return np.array(di + tri, dtype=float)


def extract_features(seq: str, target_len: int) -> np.ndarray:
    """Full feature vector used by the subtype and pathogenicity models:
    positional chemical-property bits (3 x target_len) + dinuc (16) +
    trinuc (64) frequencies, computed on the length-fixed sequence."""
    seq = fix_length(seq, target_len)
    chem = chemical_property_features(seq)
    di = kmer_frequency(seq, 2, DINUCS)
    tri = kmer_frequency(seq, 3, TRINUCS)
    return np.array(chem + di + tri, dtype=float)


def feature_names(target_len: int) -> list[str]:
    names = []
    for i in range(target_len):
        names += [f"chem_ring_pos{i+1}", f"chem_amino_pos{i+1}", f"chem_hbond_pos{i+1}"]
    names += [f"dinuc_{d}" for d in DINUCS]
    names += [f"trinuc_{t}" for t in TRINUCS]
    return names
