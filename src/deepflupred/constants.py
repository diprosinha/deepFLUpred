"""Shared constants for deepFLUpred."""

PACKAGE_VERSION = "0.1.0"

SEGMENTS = ("HA", "NA")

HA_SUBTYPES = tuple(f"H{i}" for i in range(1, 17))
NA_SUBTYPES = tuple(f"N{i}" for i in range(1, 10))

# "human" is not currently supported: the DNABERT+BiLSTM pathogenicity model
# was only trained for chicken and duck cleavage-site data.
PATHOGENICITY_HOSTS = ("chicken", "duck")
PATHOGENICITY_CLASSES = ("HPAI", "LPAI")

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
