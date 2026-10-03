"""Tautomer-aware scoring, submission I/O, and structure-disjoint folds."""

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=65_536)
def structure_key(smiles: str) -> str:
    """Return the canonical-tautomer InChIKey connectivity block.

    Connectivity-block comparison intentionally ignores stereochemistry.
    RDKit diagnostics are left enabled; invalid inputs never enter the cache.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError("SMILES does not describe a molecule")
        canonical = _TAUTOMER_ENUMERATOR.Canonicalize(molecule)
        key = Chem.MolToInchiKey(canonical)
        if not key or len(key.split("-", 1)[0]) != 14:
            raise ValueError("Could not generate an InChIKey")
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid SMILES: {smiles!r}") from exc
    return key.split("-", 1)[0]


def evaluate_predictions(
    predictions: dict[str, list[str]], truth: dict[str, str]
) -> dict:
    """Score each truth molecule equally, retaining invalid prediction ranks."""
    reciprocal_rank = 0.0
    top1 = top5 = top25 = 0
    for molecule_id, smiles in truth.items():
        target = structure_key(smiles)
        for rank, candidate in enumerate(predictions.get(molecule_id, [])[:25], 1):
            try:
                predicted = structure_key(candidate)
            except ValueError:
                continue
            if predicted == target:
                reciprocal_rank += 1.0 / rank
                top1 += rank == 1
                top5 += rank <= 5
                top25 += 1
                break
    molecules = len(truth)
    denominator = molecules or 1
    return {
        "mrr_at_25": reciprocal_rank / denominator,
        "top1_accuracy": top1 / denominator,
        "top5_accuracy": top5 / denominator,
        "top25_recall": top25 / denominator,
        "molecules": molecules,
    }


def _id_set(molecule_ids: list[str]) -> set[str]:
    if any(not isinstance(value, str) or not value.strip() for value in molecule_ids):
        raise ValueError("Molecule IDs must be nonempty strings")
    result = set(molecule_ids)
    if len(result) != len(molecule_ids):
        raise ValueError("Molecule IDs must be unique")
    return result


def _check_coverage(actual: set[str], expected: set[str]) -> None:
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Submission ID mismatch: {len(missing)} missing {missing[:5]!r}; "
            f"{len(extra)} unexpected {extra[:5]!r}"
        )


def _candidates(values: list[str], molecule_id: str, *, deduplicate: bool) -> list[str]:
    if not isinstance(values, list) or not 1 <= len(values) <= 25:
        raise ValueError(f"{molecule_id!r} must have 1..25 SMILES candidates")
    result = []
    seen = set()
    for rank, smiles in enumerate(values, 1):
        if not isinstance(smiles, str) or not smiles.strip() or ";" in smiles:
            raise ValueError(f"Invalid candidate at {molecule_id!r}, rank {rank}")
        try:
            key = structure_key(smiles)
        except ValueError as exc:
            raise ValueError(
                f"Invalid SMILES at {molecule_id!r}, rank {rank}: {smiles!r}"
            ) from exc
        if not deduplicate or key not in seen:
            result.append(smiles)
        seen.add(key)
    return result


def _submission_summary(counts: list[int]) -> dict:
    return {
        "molecules": len(counts),
        "candidates": sum(counts),
        "min_candidates": min(counts, default=0),
        "max_candidates": max(counts, default=0),
    }


def write_submission(
    predictions: dict[str, list[str]], molecule_ids: list[str], output: Path
) -> dict:
    """Validate first, then write ordered, structurally deduplicated candidates."""
    expected = _id_set(molecule_ids)
    actual = _id_set(list(predictions))
    _check_coverage(actual, expected)
    rows = [
        (molecule_id, _candidates(predictions[molecule_id], molecule_id, deduplicate=True))
        for molecule_id in molecule_ids
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule_id", "smiles"])
        for molecule_id, candidates in rows:
            writer.writerow([molecule_id, ";".join(candidates)])
    return _submission_summary([len(candidates) for _, candidates in rows])


def validate_submission(path: Path, expected_ids: list[str]) -> dict:
    """Check a submission without changing its order or contents."""
    expected = _id_set(expected_ids)
    actual = set()
    counts = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        if next(reader, None) != ["molecule_id", "smiles"]:
            raise ValueError("Submission header must be exactly molecule_id,smiles")
        for line, row in enumerate(reader, 2):
            if len(row) != 2:
                raise ValueError(f"Submission row {line} must have exactly two columns")
            molecule_id, smiles = row
            if not molecule_id.strip():
                raise ValueError(f"Submission row {line} has an empty molecule ID")
            if molecule_id in actual:
                raise ValueError(f"Duplicate submission molecule ID: {molecule_id!r}")
            actual.add(molecule_id)
            candidates = _candidates(smiles.split(";"), molecule_id, deduplicate=False)
            counts.append(len(candidates))
    _check_coverage(actual, expected)
    return _submission_summary(counts)


def write_structure_folds(
    train_path: Path, output: Path, n_folds: int = 5, seed: int = 42
) -> dict:
    """Keep connected raw and canonical identities in one deterministic fold.

    Bounded tautomer enumeration can yield different keys for representations
    sharing a raw InChIKey14. Union those keys rather than splitting or dropping
    the spectra. The lexicographically smallest component key names its group.
    """
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    train_path = Path(train_path)
    output = Path(output)
    if train_path.resolve() == output.resolve():
        raise ValueError("Fold output must not overwrite the training parquet")

    smiles_to_key: dict[str, str] = {}
    raw_to_key: dict[str, str] = {}
    raw_variants: dict[str, set[str]] = {}
    parents: dict[str, str] = {}

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    spectra = 0
    next_progress = 500_000
    source = pq.ParquetFile(train_path)
    try:
        for batch in source.iter_batches(
            batch_size=65_536, columns=["inchikey14", "normalized_smiles"]
        ):
            raw_keys = batch.column(0).to_pylist()
            smiles_values = batch.column(1).to_pylist()
            for raw, smiles in zip(raw_keys, smiles_values, strict=True):
                spectra += 1
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError(f"Missing inchikey14 at training row {spectra}")
                if not isinstance(smiles, str) or not smiles.strip():
                    raise ValueError(f"Missing normalized_smiles at training row {spectra}")
                key = smiles_to_key.get(smiles)
                if key is None:
                    key = structure_key(smiles)
                    smiles_to_key[smiles] = key
                    parents.setdefault(key, key)
                existing = raw_to_key.get(raw)
                if existing is None:
                    raw_to_key[raw] = key
                elif existing != key:
                    raw_variants.setdefault(raw, {existing}).add(key)
                    left, right = find(existing), find(key)
                    if left != right:
                        parents[max(left, right)] = min(left, right)
            if spectra >= next_progress:
                print(f"Grouped {spectra:,} spectra / {len(raw_to_key):,} raw structures", flush=True)
                next_progress += 500_000
    finally:
        source.close()

    smiles_to_key.clear()
    group_to_fold: dict[str, int] = {}
    raw_counts = [0] * n_folds
    canonical_counts = [0] * n_folds
    for key in parents:
        group = find(key)
        fold = group_to_fold.get(group)
        if fold is None:
            digest = hashlib.sha256(f"{seed}:{group}".encode("ascii")).digest()
            fold = int.from_bytes(digest, "big") % n_folds
            group_to_fold[group] = fold
        canonical_counts[fold] += 1

    schema = pa.schema([
        ("inchikey14", pa.string()),
        ("structure_keys", pa.list_(pa.string())),
        ("split_group", pa.string()),
        ("fold", pa.int32()),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        raw_batch: list[str] = []
        key_batch: list[list[str]] = []
        group_batch: list[str] = []
        fold_batch: list[int] = []
        for raw, key in raw_to_key.items():
            group = find(key)
            fold = group_to_fold[group]
            raw_counts[fold] += 1
            raw_batch.append(raw)
            key_batch.append(sorted(raw_variants[raw]) if raw in raw_variants else [key])
            group_batch.append(group)
            fold_batch.append(fold)
            if len(raw_batch) == 65_536:
                writer.write_table(pa.Table.from_pydict(
                    {"inchikey14": raw_batch, "structure_keys": key_batch,
                     "split_group": group_batch, "fold": fold_batch},
                    schema=schema,
                ))
                raw_batch.clear()
                key_batch.clear()
                group_batch.clear()
                fold_batch.clear()
        if raw_batch:
            writer.write_table(pa.Table.from_pydict(
                {"inchikey14": raw_batch, "structure_keys": key_batch,
                 "split_group": group_batch, "fold": fold_batch},
                schema=schema,
            ))
    return {
        "spectra": spectra,
        "raw_structures": len(raw_to_key),
        "canonical_structures": len(parents),
        "split_groups": len(group_to_fold),
        "ambiguous_raw_structures": len(raw_variants),
        "n_folds": n_folds,
        "seed": seed,
        "fold_counts": {str(fold): count for fold, count in enumerate(raw_counts)},
        "canonical_fold_counts": {
            str(fold): count for fold, count in enumerate(canonical_counts)
        },
    }
