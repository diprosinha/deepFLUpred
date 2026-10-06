"""Command-line interface for deepFLUpred."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepflupred import __version__
from deepflupred.constants import GENES, PATHOGENICITY_HOSTS
from deepflupred.model_store import ModelStore
from deepflupred.predict import predict_fasta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepflupred",
        description=(
            "deepFLUpred: a deep learning-based framework for genomic characterization, "
            "subtyping, and pathogenicity prediction of avian influenza viruses. "
            "Screens a sequence against avian influenza A, identifies which of the eight "
            "genes it is (PB2/PB1/PA/HA/NP/NA/MP/NS), predicts HA (H1-H16) / NA (N1-N9) "
            "subtype, and -- for HA -- predicts HPAI/LPAI pathogenicity from the HA1/HA2 "
            "cleavage site."
        ),
    )
    parser.add_argument("--version", action="version", version=f"deepFLUpred {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict", help="Classify sequences in a nucleotide FASTA file"
    )
    predict.add_argument("fasta", type=Path, help="Nucleotide FASTA file")
    predict.add_argument(
        "--output", "-o", type=Path, required=True, help="CSV file for results"
    )
    predict.add_argument(
        "--expected-gene",
        choices=GENES,
        help="Expected gene, when known; a different assignment is flagged, not rejected",
    )
    predict.add_argument(
        "--pathogenicity-host",
        choices=PATHOGENICITY_HOSTS,
        action="append",
        dest="pathogenicity_hosts",
        help="Restrict HPAI/LPAI prediction to this host's model "
        "(repeatable; default: run all supported hosts and report a consensus)",
    )
    predict.add_argument("--gene-model", type=Path, help="Alternative phase-2 gene-ID model")
    predict.add_argument(
        "--oneclass-model-dir", type=Path, help="Directory with an alternative phase-1 screen"
    )
    predict.add_argument(
        "--no-screen",
        action="store_true",
        help="Skip the phase-1 screen and classify every sequence (diagnostic use: "
        "without the screen, a non-influenza input is still forced into a gene call)",
    )
    predict.add_argument(
        "--subtype-model-dir", type=Path, help="Directory with alternative HA/NA subtype models"
    )
    predict.add_argument(
        "--pathogenicity-model-dir",
        type=Path,
        help="Directory with alternative per-host pathogenicity models",
    )
    predict.add_argument("--quiet", action="store_true")

    models = subparsers.add_parser("models", help="Show the bundled models")
    models.add_argument("--json", action="store_true", help="Write the model list as JSON")
    models.add_argument("--verify", action="store_true", help="Verify the bundled model files")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "models":
        if args.verify:
            verification = ModelStore.verify_bundled_models()
            if args.json:
                print(json.dumps(verification, indent=2, sort_keys=True))
            else:
                for path, valid in verification.items():
                    print(f"{'Verified' if valid else 'Failed'}\t{path}")
            if not all(verification.values()):
                raise SystemExit(1)
            return
        inventory = ModelStore().inventory()
        if args.json:
            print(json.dumps(inventory, indent=2, sort_keys=True))
        else:
            for row in inventory:
                print(f"{row['model']}\t{row['task']}")
                m = row.get("test_metrics")
                if m:
                    parts = [
                        f"{key}={value:.4f}" if isinstance(value, (int, float)) else f"{key}={value}"
                        for key, value in m.items()
                    ]
                    print("  " + "  ".join(parts))
        return

    table = predict_fasta(
        args.fasta,
        args.output,
        expected_gene=args.expected_gene,
        pathogenicity_hosts=args.pathogenicity_hosts,
        gene_model=args.gene_model,
        subtype_model_dir=args.subtype_model_dir,
        pathogenicity_model_dir=args.pathogenicity_model_dir,
        oneclass_model_dir=args.oneclass_model_dir,
        run_screen=not args.no_screen,
    )
    if not args.quiet:
        for _, result in table.iterrows():
            print(f"{result['sequence_id']}: {result['status']}")
            print(f"  {result['explanation']}")
        print(f"\nResults CSV: {args.output.resolve()}")
