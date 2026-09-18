from flumapper.cleavage_site import analyze_sequence


def test_no_motif_returns_none():
    assert analyze_sequence("ACGT" * 5) is None


def test_polybasic_motif_detected():
    # Frame 0: AAA AAA (K K) ... then a polybasic run RRR RKR before GLFG
    # (G=GGA, L=CTG, F=TTC, G=GGA), padded so the window is fully covered.
    upstream = "A" * 200
    basic = "CGT CGT CGT AAA CGT AGA".replace(" ", "")  # R R R K R R
    motif = "GGACTGTTCGGA"  # G L F G
    downstream = "A" * 200
    seq = upstream + basic + motif + downstream
    result = analyze_sequence(seq)
    assert result is not None
    assert result["classification"] == "Polybasic"
    assert result["basic_residues_in_last10"] >= 4
    assert result["window_seq"] is not None
    assert len(result["window_seq"]) == 110
