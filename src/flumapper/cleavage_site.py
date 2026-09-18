"""HA1/HA2 cleavage-site localization and rule-based Polybasic/Monobasic call.

Ported from gisaid_data/hpli_lpai/preprocess_cleavage_site.py (the pipeline
used to build the training data for the pathogenicity models) so FluMAPPER
can locate and extract the same 110nt ML window from an arbitrary full-length
HA nucleotide sequence at inference time.

Method
------
1. Translate the sequence in reading frames 0, 1, 2 (in that order), keeping
   in-frame stop codons as '*' rather than truncating (raw submissions often
   carry a few extra 5' nt before the true ATG).
2. Search for the conserved HA2 fusion-peptide start motif G[ILV][FY][GAS]
   (covers GLFG/GIFG/GLFS). The first frame with a hit is used; the leftmost
   hit in that frame is the HA1/HA2 boundary.
3. The up to 10 residues immediately preceding the motif are the
   cleavage-site motif; R/K residues in it are counted.
     - any ambiguous residue ('X'/'*') in the motif -> Ambiguous/Atypical
     - >=4 basic residues  -> Polybasic  (the WHO/OIE HPAI-type signature)
     - <=3 basic residues  -> Monobasic  (LPAI-type)
     - no motif found in any frame -> Not resolved
4. A fixed 110nt window (69nt upstream + 40nt downstream of the last
   nucleotide of the HA1 codon) is extracted as the ML feature window fed to
   the pathogenicity models -- identical to the window used to build their
   training data.
"""
from __future__ import annotations

STOP_CODONS = {"TAA", "TAG", "TGA"}

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

import re

MOTIF_RE = re.compile(r"G[ILV][FY][GAS]")


def translate_frame(seq: str, frame: int) -> tuple[str, list[tuple[int, int]]]:
    aa: list[str] = []
    coords: list[tuple[int, int]] = []
    i = frame
    n = len(seq)
    while i + 3 <= n:
        codon = seq[i : i + 3]
        if codon in STOP_CODONS:
            residue = "*"
        else:
            residue = CODON_TABLE.get(codon, "X") if set(codon) <= set("ACGT") else "X"
        aa.append(residue)
        coords.append((i + 1, i + 3))  # 1-based inclusive
        i += 3
    return "".join(aa), coords


def analyze_sequence(seq: str) -> dict | None:
    """Return a dict describing the HA1/HA2 cleavage site (including the
    110nt `window_seq` ML window when the full window is covered by the
    input sequence), or None if no cleavage-site motif is found in any
    reading frame."""
    seq = seq.upper().replace("U", "T")
    for frame in (0, 1, 2):
        aa_seq, coords = translate_frame(seq, frame)
        m = MOTIF_RE.search(aa_seq)
        if not m:
            continue
        p0 = m.start()
        n_pre = min(10, p0)
        motif_aa = aa_seq[p0 - n_pre : p0]
        basic_idx = [p0 - n_pre + k for k, a in enumerate(motif_aa) if a in "RK"]
        basic_count = len(basic_idx)
        has_ambig = "X" in motif_aa or "*" in motif_aa

        if not motif_aa:
            classification = "Not resolved (cleavage site not covered by sequence / ambiguous bases)"
        elif has_ambig:
            classification = "Ambiguous/Atypical"
        elif basic_count >= 4:
            classification = "Polybasic"
        else:
            classification = "Monobasic"

        motif_nt_end = coords[p0 - 1][1] if p0 > 0 else None

        window_seq = window_start = window_end = None
        if motif_nt_end is not None:
            window_start = motif_nt_end - 69
            window_end = motif_nt_end + 40
            if window_start >= 1 and window_end <= len(seq):
                window_seq = seq[window_start - 1 : window_end]

        return {
            "classification": classification,
            "motif_aa": motif_aa,
            "basic_residues_in_last10": basic_count,
            "motif_nt_end": motif_nt_end,
            "reading_frame_used": frame,
            "window_start": window_start,
            "window_end": window_end,
            "window_seq": window_seq,
        }
    return None
