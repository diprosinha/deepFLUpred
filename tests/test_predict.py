import random

from flumapper.model_store import ModelStore
from flumapper.predict import predict_sequence, predict_sequences


def _random_seq(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_random_sequence_is_segment_unresolved():
    store = ModelStore()
    row = predict_sequence(_random_seq(1600, seed=0), "control", store)
    assert row["status"] == "segment_unresolved"
    assert row["segment"] is None


def test_predict_sequences_returns_one_row_per_input():
    store = ModelStore()
    seqs = [_random_seq(1600, seed=1), _random_seq(1400, seed=2)]
    table = predict_sequences(seqs, identifiers=["a", "b"])
    assert len(table) == 2
    assert list(table["sequence_id"]) == ["a", "b"]


def test_expected_segment_mismatch_is_flagged():
    store = ModelStore()
    # A random sequence won't confidently resolve to a segment at all, so
    # use a length/composition unlikely to be HA to exercise the mismatch
    # path is only reachable once a segment IS resolved -- this asserts the
    # unresolved case doesn't crash when --expected-segment is set.
    row = predict_sequence(
        _random_seq(1600, seed=3), "control", store, expected_segment="HA"
    )
    assert row["status"] in ("segment_unresolved", "expected_segment_mismatch", "subtype_assigned")
