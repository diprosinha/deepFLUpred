import random
from pathlib import Path

from deepflupred.model_store import ModelStore
from deepflupred.predict import parse_fasta, predict_sequence, predict_sequences

# Real GISAID sequences (H5 HA and N1 NA), stored lowercase in the fixture
# files to regression-test the DNABERT tokenizer case-normalization fix --
# the tokenizer is case-sensitive, so an un-normalized lowercase sequence
# used to silently produce garbage embeddings and a wrong classification.
FIXTURES = Path(__file__).parent / "fixtures"
HA_SEQ_H5 = parse_fasta(FIXTURES / "sample_ha_h5.fasta")[0][1]
NA_SEQ_N1 = parse_fasta(FIXTURES / "sample_na_n1.fasta")[0][1]


def _random_seq(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_lowercase_ha_sequence_is_correctly_classified():
    assert HA_SEQ_H5[:5].islower()
    store = ModelStore()
    row = predict_sequence(HA_SEQ_H5, "ha_query", store)
    assert row["segment"] == "HA"
    assert row["subtype"] == "H5"


def test_lowercase_na_sequence_is_correctly_classified():
    assert NA_SEQ_N1[:5].islower()
    store = ModelStore()
    row = predict_sequence(NA_SEQ_N1, "na_query", store)
    assert row["segment"] == "NA"
    assert row["subtype"] == "N1"


def test_random_sequence_does_not_crash():
    # The segment model is a forced binary HA-vs-NA classifier with no true
    # "neither" class (see README: "Segment unresolved" is a coarse
    # confidence/length gate, not a calibrated novelty detector), so a
    # specific resolved/unresolved outcome for pure random noise is not
    # guaranteed -- only that the pipeline runs and returns a valid status.
    store = ModelStore()
    row = predict_sequence(_random_seq(1600, seed=0), "control", store)
    assert row["status"] in ("segment_unresolved", "subtype_assigned")


def test_predict_sequences_returns_one_row_per_input():
    store = ModelStore()
    seqs = [_random_seq(1600, seed=1), _random_seq(1400, seed=2)]
    table = predict_sequences(seqs, identifiers=["a", "b"])
    assert len(table) == 2
    assert list(table["sequence_id"]) == ["a", "b"]


def test_expected_segment_mismatch_is_flagged():
    store = ModelStore()
    row = predict_sequence(NA_SEQ_N1, "na_query", store, expected_segment="HA")
    assert row["status"] == "expected_segment_mismatch"
    assert row["segment"] == "NA"
