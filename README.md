# deepFLUpred

**A Deep Learning-Based Framework for Genomic Characterization, Subtyping, and Pathogenicity Prediction of Avian Influenza Viruses**

Authors: Dipro Sinha, Naveen Duhan, Jagathiswaran Radhakrisnan, Sunil Mor

deepFLUpred classifies avian influenza nucleotide sequences with a
DNABERT + BiLSTM pipeline, in four gated phases:

1. **Avian influenza A screen** — a one-class BiLSTM, trained on all eight
   avian influenza A genes (positives only), that turns away sequences unlike
   avian influenza A before any classifier runs.
2. **Gene identification** — which of the eight genes is it?
   PB2, PB1, PA, HA, NP, NA, MP or NS. For the six internal segments this is the
   final answer: influenza subtyping is defined only for HA and NA.
3. **Subtype classification** — if HA: H1–H16; if NA: N1–N9.
4. **Pathogenicity prediction** — for HA only: locates the HA1/HA2 cleavage
   site, reports the biological Polybasic/Monobasic rule (the WHO/OIE
   HPAI-type signature), and predicts HPAI vs LPAI with per-host
   (chicken/duck) BiLSTM models trained on the 110nt cleavage-site window.

Each phase gates the next: the classifiers only run on sequences the screen
accepts, subtype classification only runs for HA or NA, and pathogenicity
prediction only runs for HA. Prediction is alignment-free and does not run BLAST
or pairwise identity/alignment.

The screen exists because phase 2 is a forced eight-way choice with no "neither"
class: without it, random noise comes back with a gene. A screen rejection means
"unlike avian influenza A" — it cannot identify what a rejected sequence
actually is. It reports its score, its threshold, and the margin between them,
so a borderline call is visible as one.

Every phase shares the same feature pipeline:

```
DNA sequence -> DNABERT tokenizer -> DNABERT-2 Transformer (frozen)
  -> per-token hidden states -> adaptive average-pooling to a fixed
     16-step sequence -> BiLSTM (trained per phase)
```

Phases 1, 2 and 3 all consume an identical encoding of the full input sequence,
so that embedding is computed once per sequence and reused across all three
rather than re-running the transformer three times. Phase 4 encodes only the
extracted 110nt cleavage-site window.

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

If the expected gene is known, provide it as an additional check (a
different assignment is flagged, not rejected):

```bash
deepflupred predict query.fasta --expected-gene PB2 --output predictions.csv
```

Restrict HPAI/LPAI prediction to a specific host's model instead of the
default chicken+duck consensus:

```bash
deepflupred predict query.fasta --pathogenicity-host chicken --output predictions.csv
```

The output CSV has one row per sequence with the screen's verdict, the gene call
and confidence, and — for HA and NA — the subtype call and confidence, plus,
for HA, the cleavage-site classification/motif and the majority-vote HPAI/LPAI
consensus across host models.

> **Note on reading the CSV with pandas**: the `gene` column's value for a
> neuraminidase call is the literal string `"NA"`, which pandas' `read_csv`
> treats as a missing-value marker by default and silently turns into `NaN`.
> Read the results with `pd.read_csv(path, keep_default_na=False)`
> (or `dtype=str`) to preserve it.

## Prediction status

- **not_avian_influenza** — rejected by the phase-1 screen; no gene or subtype
  was assigned. Check the reported margin before concluding a sequence is not
  influenza.
- **gene_identified** — phase 2 named the gene, and it is one of the six
  internal segments (PB2/PB1/PA/NP/MP/NS). This is a complete answer, not a
  partial one: subtype and pathogenicity are defined only for HA and NA.
- **subtype_assigned** — gene (HA or NA) and subtype were both assigned; for HA,
  pathogenicity was attempted too.
- **gene_unresolved** — the sequence passed the screen but no gene reached the
  75% confidence floor. The best guess and the runner-up are reported in the
  explanation, so a near-tie is visible.
- **expected_gene_mismatch** — the assigned gene differs from `--expected-gene`.
- **Cleavage site not resolved** — for an HA sequence, no HA1/HA2 cleavage-site
  motif (`G[ILV][FY][GAS]`) was found in any reading frame, or the 110nt ML
  window was not fully covered by the input sequence; pathogenicity was not
  assessed.

## Python API

```python
from deepflupred import predict_fasta, predict_sequences

table = predict_fasta("query.fasta", output="predictions.csv", expected_gene="HA")

table = predict_sequences(["ACGT..."], identifiers=["query-1"])
```

## Models

deepFLUpred bundles the following, all sharing the frozen DNABERT-2 encoder
described above with a BiLSTM head trained per phase:

- **one phase-1 screen** (avian influenza A, any gene) — a BiLSTM autoencoder
  trained on positives only across all eight segments, threshold calibrated to
  accept 95% of its training distribution. Held-out acceptance of genuine avian
  influenza A is 94.9%; it rejects 98.3% of infectious bronchitis virus, 96.3%
  of Newcastle disease virus and 95.6% of chicken host transcripts, and accepts
  97.6% of avian IAV internal segments drawn from NCBI (a different database
  than it was trained on). See `../new_analysis/phase12/`.
- **one phase-2 gene classifier** (8 classes: PB2/PB1/PA/HA/NP/NA/MP/NS),
  trained on 99%-identity cd-hit cluster representatives from a GISAID EpiFlu
  avian pull, split 90:10 cluster-disjoint — 100% accuracy and macro-F1 on
  4,695 held-out sequences, with no confusion between any pair of genes
- **two subtype classifiers** (HA: H1–H16, NA: N1–N9), each fit on
  99%-identity cluster-disjoint GISAID data (see the model-development
  pipeline in `../HA_gene`, `../NA_gene`, and `../new_analysis`)
- **two pathogenicity classifiers** (HPAI vs LPAI), one per host
  (chicken/duck), each trained on 110nt HA1/HA2 cleavage-site windows (see
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

## Known limits

Measured, not hypothetical. Each is quantified in
`../new_analysis/phase12/results/phase1/`.

- **Phase 1 does not discriminate host.** It accepts 87.3% of human-host
  influenza A HA/NA. A sequence that passes is "influenza A-like", not
  "confirmed avian origin".
- **PB2 is the screen's weakest gene** at 89.0% held-out acceptance (PA 92.1%,
  PB1 93.0%, HA 93.6%; MP and NS are above 99.7%). Roughly one genuine PB2 in
  nine is turned away. When that happens it is the *screen* being conservative:
  the phase-2 model still identifies those same sequences as PB2 correctly if
  you pass `--no-screen`. Check the reported margin on any surprising rejection.
- **Influenza B is rejected only 64.4% of the time** (n=59). Influenza B genes
  are homologous to influenza A's, and 0.2.0 has one screen where 0.1.0 had two
  in series — the removed second gate was contributing to this. If you expect
  influenza B in your input, do not rely on the screen to remove it.
- **The screen's margin is thin.** Tightening from the shipped 95% nominal rate
  to 98% collapses NDV rejection from 96.3% to 26.1%, while loosening to 90%
  costs PB2 acceptance (89.0% → 76.8%) for only a small rejection gain. The 95%
  point is close to the best available trade-off, and
  `operating_characteristics.csv` tabulates the alternatives per gene. Both the
  Python API and the CLI accept a threshold override, so changing the operating
  point does not require retraining.
- **Phase 2's 100% held-out accuracy reflects an easy task**, not an unusually
  strong model: the eight segments are unrelated genes. It is not inflated by the
  split (cluster-disjoint at 99% identity) nor by length (PB2 and PB1 are both
  2,341 nt and are never confused), but it does mean the realistic failure mode
  at inference is out-of-distribution input, which is phase 1's job rather than
  phase 2's.
- **No human-host pathogenicity model** exists for phase 4 (chicken and duck
  only).

## What changed in 0.2.0

Before 0.2.0 the pipeline had five stages built around a binary HA-vs-NA
segment classifier, with two one-class gates in front of it:

| 0.1.0 | 0.2.0 |
| --- | --- |
| stage 0: OneClassSVM, "is this influenza?" | phase 1: one-class BiLSTM, all eight genes |
| stage 1: one-class BiLSTM, "is this H or N?" | *removed — phase 2 answers it* |
| stage 2: HA-vs-NA binary classifier | phase 2: eight-class gene classifier |
| stage 3: subtype | phase 3: subtype (unchanged) |
| stage 4: pathogenicity | phase 4: pathogenicity (unchanged) |

The old stage 1 existed only because the segment classifier could not say
"neither" — handed a PB2 it had to answer HA or NA. The gate could reject such a
sequence but never explain it: the output was "not an H or N gene", with no way
to learn it was PB2. Phase 2 names it instead. Two measured weaknesses of the old
gates also go away: stage 0 rejected 10–17% of *genuine* PA/PB1/PB2 segments, and
stage 1 caught NP only 51.2% of the time.

API changes: `--expected-segment` → `--expected-gene` (and now accepts all eight
genes), the `segment` output column → `gene`, `predict_*(expected_segment=...)` →
`expected_gene=...`, `run_gates=` → `run_screen=`, and `ModelStore(segment_model=)`
→ `ModelStore(gene_model=)`.

## Citation

Citation metadata are available in `CITATION.cff`.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```
