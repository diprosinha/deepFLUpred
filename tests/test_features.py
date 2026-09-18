import numpy as np

from flumapper.features import extract_features, fix_length, kmer_only_features


def test_fix_length_pads_with_n():
    assert fix_length("ACGT", 8) == "ACGTNNNN"


def test_fix_length_truncates():
    assert fix_length("ACGTACGTACGT", 4) == "ACGT"


def test_kmer_only_features_length_invariant_dimension():
    short = kmer_only_features("ACGTACGTACGT")
    long = kmer_only_features("ACGT" * 200)
    assert short.shape == long.shape == (80,)  # 16 dinuc + 64 trinuc


def test_kmer_only_features_sums_to_two():
    # dinuc frequencies sum to 1, trinuc frequencies sum to 1 -> total 2
    feats = kmer_only_features("ACGT" * 50)
    assert np.isclose(feats.sum(), 2.0, atol=1e-6)


def test_extract_features_dimension():
    feats = extract_features("ACGT" * 50, target_len=200)
    assert feats.shape == (200 * 3 + 16 + 64,)
