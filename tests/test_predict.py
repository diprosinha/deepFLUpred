import random
from pathlib import Path

from deepflupred.model_store import ModelStore
from deepflupred.predict import parse_fasta, predict_sequence, predict_sequences

# Real GISAID sequences, stored lowercase in the fixture files to regression-test
# the DNABERT tokenizer case-normalization fix -- the tokenizer is case-sensitive,
# so an un-normalized lowercase sequence used to silently produce garbage
# embeddings and a wrong classification.
#
# The PB2 and MP fixtures are drawn from the phase-1/phase-2 HELD-OUT split
# (new_analysis/phase12/data/splits/phase1_test_positives.fasta), so they are
# cluster-disjoint from everything the gene model was trained on.
FIXTURES = Path(__file__).parent / "fixtures"
HA_SEQ_H5 = parse_fasta(FIXTURES / "sample_ha_h5.fasta")[0][1]
NA_SEQ_N1 = parse_fasta(FIXTURES / "sample_na_n1.fasta")[0][1]
PB2_SEQ = parse_fasta(FIXTURES / "sample_pb2.fasta")[0][1]
MP_SEQ = parse_fasta(FIXTURES / "sample_mp.fasta")[0][1]
# A real held-out PB2 that the phase-1 screen turns away. PB2 is the screen's
# weakest gene (89.0% held-out acceptance), so this fixture pins that known
# blind spot rather than leaving it to be rediscovered.
PB2_SEQ_SCREEN_REJECTED = parse_fasta(FIXTURES / "sample_pb2_screen_rejected.fasta")[0][1]


def _random_seq(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_lowercase_ha_sequence_is_correctly_classified():
    assert HA_SEQ_H5[:5].islower()
    store = ModelStore()
    row = predict_sequence(HA_SEQ_H5, "ha_query", store)
    assert row["gene"] == "HA"
    assert row["subtype"] == "H5"


def test_lowercase_na_sequence_is_correctly_classified():
    assert NA_SEQ_N1[:5].islower()
    store = ModelStore()
    row = predict_sequence(NA_SEQ_N1, "na_query", store)
    assert row["gene"] == "NA"
    assert row["subtype"] == "N1"


def test_internal_segment_is_named_not_forced_into_ha_or_na():
    # The pre-0.2.0 pipeline could only answer "HA" or "NA", so an internal
    # segment was either misassigned or rejected by a gate that could not say
    # what it was. Phase 2 names it, and stops there: subtyping is HA/NA only.
    store = ModelStore()
    for sequence, expected in ((PB2_SEQ, "PB2"), (MP_SEQ, "MP")):
        row = predict_sequence(sequence, f"{expected.lower()}_query", store)
        assert row["gene"] == expected
        assert row["status"] == "gene_identified"
        assert "subtype" not in row
        assert "pathogenicity_consensus" not in row


def test_internal_segments_pass_the_phase1_screen():
    # Phase 1 is trained on all eight genes, so an internal segment is an
    # in-distribution positive -- not a novelty, as it was for the old H/N gate.
    store = ModelStore()
    for sequence in (PB2_SEQ, MP_SEQ):
        row = predict_sequence(sequence, "internal", store)
        assert row["avian_influenza_decision"] == "accepted"
        assert row["avian_influenza_margin"] >= 0


def test_screen_rejects_some_genuine_pb2_but_the_gene_model_does_not():
    # The screen's documented cost: it accepts only 89.0% of held-out PB2, so a
    # real PB2 can be turned away. What must stay true is that this is the
    # SCREEN's conservatism and not a phase-2 failure -- with the screen
    # bypassed, the gene model still calls the same sequence PB2. If a retrained
    # screen ever accepts this sequence, that is an improvement: update the
    # fixture rather than loosening the assertion.
    store = ModelStore()
    screened = predict_sequence(PB2_SEQ_SCREEN_REJECTED, "pb2_edge", store)
    assert screened["status"] == "not_avian_influenza"
    assert screened["avian_influenza_margin"] < 0

    unscreened = predict_sequence(PB2_SEQ_SCREEN_REJECTED, "pb2_edge", store, run_screen=False)
    assert unscreened["gene"] == "PB2"


def test_random_sequence_is_rejected_by_the_one_class_screen():
    # The gene model is a forced eight-way choice with no "neither" class, so
    # random noise would otherwise come back with a gene. The phase-1 screen
    # runs first and turns it away instead.
    store = ModelStore()
    row = predict_sequence(_random_seq(1600, seed=0), "control", store)
    assert row["status"] == "not_avian_influenza"
    assert row["avian_influenza_decision"] == "rejected"
    assert row["avian_influenza_margin"] < 0
    # Rejected at phase 1 means the later phases never ran.
    assert "gene" not in row
    assert "subtype" not in row


def test_screen_can_be_skipped():
    # run_screen=False restores the pre-screen behaviour, so the classifiers
    # remain reachable on their own for diagnostics.
    store = ModelStore()
    row = predict_sequence(_random_seq(1600, seed=0), "control", store, run_screen=False)
    assert row["status"] in ("gene_unresolved", "gene_identified", "subtype_assigned")
    assert "avian_influenza_decision" not in row


def test_real_sequences_pass_the_screen():
    store = ModelStore()
    for sequence, identifier in ((HA_SEQ_H5, "ha_query"), (NA_SEQ_N1, "na_query")):
        row = predict_sequence(sequence, identifier, store)
        assert row["avian_influenza_decision"] == "accepted"


def test_predict_sequences_returns_one_row_per_input():
    # Screened-out sequences still get a row, so a batch never loses inputs.
    store = ModelStore()
    seqs = [_random_seq(1600, seed=1), _random_seq(1400, seed=2)]
    table = predict_sequences(seqs, identifiers=["a", "b"])
    assert len(table) == 2
    assert list(table["sequence_id"]) == ["a", "b"]


def test_expected_gene_mismatch_is_flagged():
    store = ModelStore()
    row = predict_sequence(NA_SEQ_N1, "na_query", store, expected_gene="HA")
    assert row["status"] == "expected_gene_mismatch"
    assert row["gene"] == "NA"


def test_expected_gene_accepts_an_internal_segment():
    # --expected-gene now spans all eight genes, not just HA/NA.
    store = ModelStore()
    row = predict_sequence(PB2_SEQ, "pb2_query", store, expected_gene="PB2")
    assert row["status"] == "gene_identified"
    assert row["gene"] == "PB2"
