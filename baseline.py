"""Local/offline CASMI spectral retrieval, validation and submission commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from evaluation import (
    evaluate_predictions,
    validate_submission,
    write_structure_folds,
    write_submission,
)
from retrieval import search_library

ROOT = Path(__file__).resolve().parent


def save_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {path}", flush=True)


def value_counts(column: pa.ChunkedArray) -> dict:
    return {
        str(item["values"]): item["counts"]
        for item in pc.value_counts(column).to_pylist()
    }


def audit(data_dir: Path, output: Path) -> None:
    train = pq.ParquetFile(data_dir / "train.parquet")
    test = pq.read_table(data_dir / "test.parquet")
    metadata = train.read(
        columns=["inchikey14", "ingest_lib", "ionization_mode", "adduct"]
    )
    result = {
        "train": {
            "spectra": train.metadata.num_rows,
            "structures": pc.count_distinct(metadata["inchikey14"]).as_py(),
            "columns": train.schema_arrow.names,
            "null_structure_keys": metadata["inchikey14"].null_count,
            **{
                name: value_counts(metadata[name])
                for name in ("ingest_lib", "ionization_mode", "adduct")
            },
        },
        "test": {
            "spectra": test.num_rows,
            "molecules": pc.count_distinct(test["molecule_id"]).as_py(),
            "columns": test.column_names,
            "precursor_mz_range": pc.min_max(test["precursor_mz"]).as_py(),
            "peaks_min_p25_p50_p75_max": np.percentile(
                pc.list_value_length(test["ms2_mzs"]).to_numpy(),
                [0, 25, 50, 75, 100],
            ).tolist(),
            **{
                name: value_counts(test[name])
                for name in ("ionization_mode", "adduct", "instrument_type")
            },
        },
        "visible_test_notice": (
            "The competition's visible test contains training examples. "
            "Use it for pipeline checks, not generalization scores."
        ),
    }
    save_json(output, result)
    print(
        f"Train: {result['train']['spectra']:,} spectra / "
        f"{result['train']['structures']:,} structures; "
        f"test: {result['test']['spectra']:,} spectra / "
        f"{result['test']['molecules']} molecules",
        flush=True,
    )


def search_options(args: argparse.Namespace) -> dict:
    return {
        "ppm": args.ppm,
        "peak_tolerance": args.peak_tolerance,
        "max_peaks": args.max_peaks,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
    }


def predict(args: argparse.Namespace) -> None:
    queries = pq.read_table(args.data_dir / "test.parquet")
    molecule_ids = sorted(pc.unique(queries["molecule_id"]).to_pylist())
    predictions, statistics = search_library(
        args.data_dir / "train.parquet", queries, **search_options(args)
    )
    summary = write_submission(predictions, molecule_ids, args.output)
    save_json(args.report, {"submission": summary, "retrieval": statistics})
    print(json.dumps(summary, indent=2), flush=True)


def validate(args: argparse.Namespace) -> None:
    train_path = args.data_dir / "train.parquet"
    held_out = ds.dataset(train_path, format="parquet").to_table(
        filter=ds.field("ingest_lib") == args.library
    )
    if not held_out.num_rows:
        raise ValueError(f"No spectra found for held-out library {args.library!r}")
    keys = held_out["inchikey14"].to_pylist()
    labels = held_out["normalized_smiles"].to_pylist()
    truth = {}
    for key, smiles in zip(keys, labels):
        truth.setdefault(f"validation_{key}", smiles)
    # Retrieval receives no structure labels; molecule_id is used only for grouping.
    query_columns = [
        "ms2_mzs", "ms2_normalized_intensities", "precursor_mz",
        "adduct", "ionization_mode",
    ]
    queries = held_out.select(query_columns)
    queries = queries.append_column(
        "molecule_id", pa.array([f"validation_{key}" for key in keys])
    ).append_column(
        "spectrum_id", pa.array([f"validation_spectrum_{i}" for i in range(len(keys))])
    )
    predictions, statistics = search_library(
        train_path, queries, exclude_libraries={args.library}, **search_options(args)
    )
    metrics = evaluate_predictions(predictions, truth)
    result = {
        "protocol": "Leave one complete spectral library out of retrieval",
        "held_out_library": args.library,
        "held_out_spectra": held_out.num_rows,
        "interpretation": (
            "Known-library retrieval benchmark, not structure-disjoint generalization. "
            "A held-out molecule can have independent reference spectra in other libraries."
        ),
        "metrics": metrics,
        "retrieval": statistics,
    }
    save_json(args.output_dir / "validation_metrics.json", result)
    save_json(args.output_dir / "validation_predictions.json", predictions)
    print(json.dumps(metrics, indent=2), flush=True)


def add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--peak-tolerance", type=float, default=0.02)
    parser.add_argument("--max-peaks", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8192)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datas")
    commands = parser.add_subparsers(dest="command", required=True)

    audit_parser = commands.add_parser("audit", help="Summarize actual Parquet data")
    audit_parser.add_argument("--output", type=Path, default=ROOT / "artifacts/data_audit.json")

    folds = commands.add_parser("folds", help="Build tautomer-aware structure-disjoint folds")
    folds.add_argument("--output", type=Path, default=ROOT / "artifacts/structure_folds.parquet")
    folds.add_argument("--n-folds", type=int, default=5)
    folds.add_argument("--seed", type=int, default=42)

    validation = commands.add_parser("validate", help="Measure cross-library retrieval")
    validation.add_argument("--library", default="enveda-np-examples")
    validation.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    add_search_arguments(validation)

    prediction = commands.add_parser("predict", help="Create submission.csv from test spectra")
    prediction.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    prediction.add_argument("--report", type=Path, default=ROOT / "artifacts/prediction_report.json")
    add_search_arguments(prediction)

    check = commands.add_parser("check", help="Validate submission IDs and chemical structures")
    check.add_argument("submission", type=Path)

    args = parser.parse_args()
    if args.command == "audit":
        audit(args.data_dir, args.output)
    elif args.command == "folds":
        summary = write_structure_folds(
            args.data_dir / "train.parquet", args.output, args.n_folds, args.seed
        )
        save_json(args.output.with_suffix(".json"), summary)
        print(json.dumps(summary, indent=2), flush=True)
    elif args.command == "validate":
        validate(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "check":
        test_ids = pq.read_table(args.data_dir / "test.parquet", columns=["molecule_id"])
        summary = validate_submission(
            args.submission, sorted(pc.unique(test_ids["molecule_id"]).to_pylist())
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
