"""Shared constants for deepFLUpred."""

PACKAGE_VERSION = "0.1.0"

SEGMENTS = ("HA", "NA")

HA_SUBTYPES = tuple(f"H{i}" for i in range(1, 17))
NA_SUBTYPES = tuple(f"N{i}" for i in range(1, 10))

PATHOGENICITY_HOSTS = ("chicken", "duck", "human")
PATHOGENICITY_CLASSES = ("HPAI", "LPAI")
# The bundled pathogenicity models were trained with integer labels (see
# gisaid_data/hpli_lpai/*/train.py: display_labels=["LPAI (0)", "HPAI (1)"]).
PATHOGENICITY_LABEL_MAP = {0: "LPAI", 1: "HPAI"}

# Fixed padded/truncated length used at training time for each model family
# (must match exactly what the bundled model was trained on).
SEGMENT_TARGET_LEN = None  # segment ID uses length-invariant k-mer features only
HA_TARGET_LEN = 1800
NA_TARGET_LEN = 1500
PATHOGENICITY_TARGET_LEN = 110

# Plausible raw-sequence length range for a candidate HA/NA segment, used as
# a sanity gate alongside model confidence when deciding "present or not".
HA_LENGTH_RANGE = (900, 2200)
NA_LENGTH_RANGE = (900, 2000)

STATUS_LABELS = {
    "segment_identified": "Segment identified",
    "segment_unresolved": "Not confidently identified as HA or NA",
    "subtype_assigned": "Subtype assigned",
    "cleavage_site_unresolved": "Cleavage site not resolved",
}
