"""Shared constants for deepFLUpred."""

PACKAGE_VERSION = "0.2.0"

# Phase 2 classes: the eight influenza A segments, in canonical segment order.
# GISAID writes the matrix segment as "MP"; deepFLUpred reports the same name.
GENES = ("PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS")

# The only two genes with a downstream subtype model (phase 3). A phase-2 call
# outside this set is a complete, terminal answer -- the gene was identified,
# and influenza subtyping is defined only for HA and NA.
SUBTYPEABLE_GENES = ("HA", "NA")

# Retained name for the phase-3 subtype model keys.
SEGMENTS = SUBTYPEABLE_GENES

HA_SUBTYPES = tuple(f"H{i}" for i in range(1, 17))
NA_SUBTYPES = tuple(f"N{i}" for i in range(1, 10))

# "human" is not currently supported: the DNABERT+BiLSTM pathogenicity model
# was only trained for chicken and duck cleavage-site data.
PATHOGENICITY_HOSTS = ("chicken", "duck")
PATHOGENICITY_CLASSES = ("HPAI", "LPAI")

# Plausible raw-sequence length window per gene, used as a sanity check
# alongside model confidence. These are the same windows phase-2 training data
# was filtered to (new_analysis/phase12/prepare_datasets.py), so the package and
# the trained model agree on what a near-full-length record looks like. They
# overlap between genes on purpose: length narrows the candidates but never
# decides the call on its own.
GENE_LENGTH_RANGES = {
    "PB2": (1650, 2700),
    "PB1": (1650, 2700),
    "PA": (1550, 2550),
    "HA": (1250, 2050),
    "NP": (1100, 1800),
    "NA": (1000, 1650),
    "MP": (720, 1180),
    "NS": (620, 1030),
}

# Kept for callers that referenced the pre-0.2.0 names.
HA_LENGTH_RANGE = GENE_LENGTH_RANGES["HA"]
NA_LENGTH_RANGE = GENE_LENGTH_RANGES["NA"]

STATUS_LABELS = {
    "not_avian_influenza": "Not avian influenza A",
    "gene_unresolved": "Gene not confidently identified",
    "gene_identified": "Gene identified (not subtyped: HA/NA only)",
    "subtype_assigned": "Subtype assigned",
    "expected_gene_mismatch": "Expected gene mismatch",
    "cleavage_site_unresolved": "Cleavage site not resolved",
}
