# deepFLUpred

**A Deep Learning-Based Framework for Genomic Characterization, Subtyping, and Pathogenicity Prediction of Avian Influenza Viruses**

Authors: Dipro Sinha, Naveen Duhan, Jagathiswaran Radhakrisnan, Sunil Mor

deepFLUpred classifies avian influenza nucleotide sequences with a
DNABERT + BiLSTM deep learning pipeline, in three gated stages:

1. **Segment identification** — is the sequence HA, NA, or neither?
2. **Subtype classification** — if HA: H1-H16; if NA: N1-N9.
3. **Pathogenicity prediction** — for HA only: locates the HA1/HA2 cleavage
   site, reports the biological Polybasic/Monobasic rule (the WHO/OIE
   HPAI-type signature), and predicts HPAI vs LPAI with per-host
   (chicken/duck) BiLSTM models trained on the 110nt cleavage-site window.

Each stage gates the next: subtype classification only runs if a segment was
confidently identified, and pathogenicity prediction only runs for sequences
classified as HA. Prediction is alignment-free and does not run BLAST or
pairwise identity/alignment.

Every stage shares the same feature pipeline:

```
DNA sequence -> DNABERT tokenizer -> DNABERT-2 Transformer (frozen)
  -> per-token hidden states -> adaptive average-pooling to a fixed
     16-step sequence -> BiLSTM classifier (trained per stage)
```

The segment and subtype stages both encode the full input sequence with an
identical recipe, so that embedding is computed once per sequence and reused
for both rather than re-running the transformer twice. Pathogenicity
prediction encodes only the extracted 110nt cleavage-site window.

> **Note**: "human" host pathogenicity prediction is not currently supported.
> The DNABERT+BiLSTM pathogenicity models were only trained on chicken and
> duck cleavage-site data (see `../new_analysis/hpli_lpai`).

## Installation

deepFLUpred requires Python 3.10 or newer and a machine that can run
PyTorch >= 2.5 (Apple Silicon MPS, CUDA, or CPU). The first `predict` call
downloads the pretrained `zhihan1996/DNABERT-2-117M` encoder from Hugging
Face (requires internet access on first use; cached locally after that).

```bash
git clone <this-repository>
cd deepFLUpred
python -m pip install .
```

## Usage

Classify one or more nucleotide sequences in a FASTA file:

```bash
deepflupred predict query.fasta --output predictions.csv
```

If the expected segment is known, provide it as an additional check (a
different assignment is flagged, not rejected):

```bash
deepflupred predict query.fasta --expected-segment HA --output predictions.csv
```

Restrict HPAI/LPAI prediction to a specific host's model instead of the
default chicken+duck consensus:

```bash
deepflupred predict query.fasta --pathogenicity-host chicken --output predictions.csv
```

The output CSV has one row per sequence with the segment call and confidence,
the subtype call and confidence, and — for HA sequences — the cleavage-site
classification/motif and the majority-vote HPAI/LPAI consensus across host
models.

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
  two-class (HA-vs-NA) gate: deepFLUpred's current models were trained only
  on HA and NA reference data, so it cannot yet positively confirm a third
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
from deepflupred import predict_fasta, predict_sequences

table = predict_fasta("query.fasta", output="predictions.csv", expected_segment="HA")

table = predict_sequences(["ACGT..."], identifiers=["query-1"])
```

## Models

deepFLUpred bundles, all sharing the frozen DNABERT-2 encoder described
above with a BiLSTM classifier head trained per stage:

- one HA-vs-NA segment classifier, trained on the cached DNABERT embeddings
  of the CD-HIT-deduplicated, cluster-disjoint 99%-identity HA and NA
  representative sequences (99.85% held-out accuracy)
- two subtype classifiers (HA: H1-H16, NA: N1-N9), each fit on the same
  99%-identity cluster-disjoint GISAID data (see the model-development
  pipeline in `../HA_gene`, `../NA_gene`, and `../new_analysis`)
- two pathogenicity classifiers (HPAI vs LPAI), one per host (chicken/duck),
  each trained on 110nt HA1/HA2 cleavage-site windows (see
  `../new_analysis/hpli_lpai`)

List the bundled models, their classes, and their held-out test performance,
or verify the bundled model files against the packaged checksum manifest:

```bash
deepflupred models
deepflupred models --verify
```

Only load replacement `.pt` models from a trusted source. PyTorch model
serialization is not safe for untrusted files.

To rebuild the bundled models from the upstream `gisaid_data/new_analysis`
pipelines (e.g. after retraining), run `python build_resources.py` from the
repository root. This requires a conda/Python environment with
`torch>=2.5`, `transformers==4.29.2`, and `einops` installed (validated in
this project's `tf_metal` conda environment).

## Citation

Citation metadata are available in `CITATION.cff`.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```
