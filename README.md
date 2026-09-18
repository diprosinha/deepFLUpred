# FluMAPPER

**An AI-based Molecular Analysis, Prediction and Profiling platform for influenza viruses**

Author: Dipro Sinha

FluMAPPER classifies avian influenza nucleotide sequences with machine
learning, in three gated stages:

1. **Segment identification** — is the sequence HA, NA, or neither?
2. **Subtype classification** — if HA: H1-H16; if NA: N1-N9.
3. **Pathogenicity prediction** — for HA only: locates the HA1/HA2 cleavage
   site, reports the biological Polybasic/Monobasic rule (the WHO/OIE
   HPAI-type signature), and predicts HPAI vs LPAI with per-host
   (chicken/duck/human) RandomForest models trained on the 110nt
   cleavage-site window.

Each stage gates the next: subtype classification only runs if a segment was
confidently identified, and pathogenicity prediction only runs for sequences
classified as HA. Prediction is alignment-free (canonical nucleotide
chemical-property and di/trinucleotide k-mer features) and does not run BLAST
or pairwise identity/alignment.

## Installation

FluMAPPER requires Python 3.10 or newer.

```bash
git clone <this-repository>
cd FluMAPPER
python -m pip install .
```

## Usage

Classify one or more nucleotide sequences in a FASTA file:

```bash
flumapper predict query.fasta --output predictions.csv
```

If the expected segment is known, provide it as an additional check (a
different assignment is flagged, not rejected):

```bash
flumapper predict query.fasta --expected-segment HA --output predictions.csv
```

Restrict HPAI/LPAI prediction to a specific host's model instead of the
default three-host consensus:

```bash
flumapper predict query.fasta --pathogenicity-host chicken --output predictions.csv
```

The output CSV has one row per sequence with the segment call and confidence,
the subtype call and confidence, and — for HA sequences — the cleavage-site
classification/motif and each host's HPAI/LPAI call plus a majority-vote
consensus.

> **Note on reading the CSV with pandas**: the `segment` column's value for
> a neuraminidase call is the literal string `"NA"`, which pandas'
> `read_csv` treats as a missing-value marker by default and silently turns
> into `NaN`. Read the results with `pd.read_csv(path, keep_default_na=False)`
> (or `dtype=str`) to preserve it.

## Prediction status

- **Segment identified / subtype assigned** — segment and subtype were both
  classified.
- **Segment unresolved** — neither HA nor NA was identified with sufficient
  confidence (below 75%) or a plausible sequence length. This is a coarse,
  two-class (HA-vs-NA) gate: FluMAPPER's current models were trained only on
  HA and NA reference data, so it cannot yet positively confirm a third
  influenza segment (PB1/PB2/PA/NP/M/NS) — it can only report that a query
  does not look confidently like HA or NA.
- **Expected segment mismatch** — the assigned segment differs from
  `--expected-segment`.
- **Cleavage site not resolved** — for an HA sequence, no HA1/HA2 cleavage-site
  motif (`G[ILV][FY][GAS]`) was found in any reading frame, or the 110nt ML
  window was not fully covered by the input sequence; pathogenicity was not
  assessed.

## Python API

```python
from flumapper import predict_fasta, predict_sequences

table = predict_fasta("query.fasta", output="predictions.csv", expected_segment="HA")

table = predict_sequences(["ACGT..."], identifiers=["query-1"])
```

## Models

FluMAPPER bundles:

- one HA-vs-NA segment classifier (RandomForest, canonical dinucleotide +
  trinucleotide frequency features — alignment-free and length-invariant)
- two subtype classifiers (HA: H1-H16, NA: N1-N9), each a RandomForest
  trained on CD-HIT-deduplicated, cluster-disjoint 99%-identity GISAID data
  (see the model-development pipeline in `../gisaid_data/HA_gene` and
  `../gisaid_data/NA_gene`)
- three pathogenicity classifiers (HPAI vs LPAI), one per host
  (chicken/duck/human), each a RandomForest trained on 110nt HA1/HA2
  cleavage-site windows (see `../gisaid_data/hpli_lpai`)

List the bundled models and their held-out test performance, or verify the
bundled model files against the packaged checksum manifest:

```bash
flumapper models
flumapper models --verify
```

Only load replacement `.joblib` models from a trusted source. Python model
serialization is not safe for untrusted files.

To rebuild the bundled models from the upstream `gisaid_data` pipelines
(e.g. after retraining), run `python build_resources.py` from the repository
root.

## Citation

Citation metadata are available in `CITATION.cff`.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```
