"""Streaming mass-filtered spectral-library retrieval, without query labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_PROTON = 1.007276466621
_WATER = 18.01056468403
_ELECTRON = 0.000548579909065
ADDUCT_SHIFTS: dict[str, float] = {
    "[M+H]+": _PROTON,
    "[M+NH4]+": 17.02654910112 + _PROTON,
    "[M-H2O+H]+": _PROTON - _WATER,
    "[M-2H2O+H]+": _PROTON - 2 * _WATER,
    "[M+Na]+": 22.9897692820 - _ELECTRON,
    "[M+K]+": 38.9637064864 - _ELECTRON,
    "[M-H]-": -_PROTON,
    "[M-H2O-H]-": -_WATER - _PROTON,
    "[M+CH2O2-H]-": 46.00547930360 - _PROTON,
    "[M+Cl]-": 34.968852682 + _ELECTRON,
}
_QUERY_COLUMNS = (
    "molecule_id", "spectrum_id", "precursor_mz", "adduct",
    "ionization_mode", "ms2_mzs", "ms2_normalized_intensities",
)
_TRAIN_COLUMNS = (
    "ingest_lib", "normalized_smiles", "inchikey14", "precursor_mz",
    "adduct", "ionization_mode", "ms2_mzs", "ms2_normalized_intensities",
)


@dataclass(slots=True)
class _Query:
    molecule_id: str
    slot: int
    neutral_mass: float
    polarity: int
    mzs: np.ndarray
    weights: np.ndarray


@dataclass(slots=True)
class _Candidate:
    smiles: str
    scores: np.ndarray
    errors: np.ndarray


def _polarity(value: object) -> int:
    if value == "positive":
        return 1
    if value == "negative":
        return -1
    return 0


def _prepare_peaks(mzs: object, intensities: object, max_peaks: int) -> tuple[np.ndarray, np.ndarray]:
    """Reject malformed spectra; retain positive peaks and normalize sqrt weights."""
    mz = np.asarray(mzs, dtype=np.float64)
    intensity = np.asarray(intensities, dtype=np.float64)
    if mz.ndim != 1 or intensity.ndim != 1 or mz.size != intensity.size or not mz.size:
        raise ValueError("peak arrays must be nonempty, one-dimensional and equally sized")
    if not np.all(np.isfinite(mz)) or not np.all(np.isfinite(intensity)):
        raise ValueError("peak arrays contain nonfinite values")
    if np.any(mz <= 0) or np.any(intensity < 0):
        raise ValueError("peak m/z must be positive and intensity nonnegative")
    positive = intensity > 0
    mz, intensity = mz[positive], intensity[positive]
    if not mz.size:
        raise ValueError("spectrum has no positive-intensity peaks")
    # Stable, explicit ties prevent platform-dependent top-peak selection.
    if mz.size > max_peaks:
        strongest = np.lexsort((mz, -intensity))[:max_peaks]
        mz, intensity = mz[strongest], intensity[strongest]
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]
    # Rescale before summing to avoid overflow with unusually large intensities.
    intensity = intensity / intensity.max()
    weights = np.sqrt(intensity / intensity.sum())
    return mz, weights


def _cosine(query: _Query, mzs: np.ndarray, weights: np.ndarray, tolerance: float) -> float:
    """Greedy largest-product one-to-one matching within an absolute Da tolerance."""
    left = np.searchsorted(mzs, query.mzs - tolerance, side="left")
    right = np.searchsorted(mzs, query.mzs + tolerance, side="right")
    counts = right - left
    total = int(counts.sum())
    if not total:
        return 0.0
    query_indices = np.repeat(np.arange(query.mzs.size), counts)
    starts = np.cumsum(counts) - counts
    reference_indices = np.repeat(left, counts) + np.arange(total) - np.repeat(starts, counts)
    products = query.weights[query_indices] * weights[reference_indices]
    if counts.max() <= 1 and np.unique(reference_indices).size == total:
        return min(1.0, float(products.sum()))
    order = np.argsort(-products, kind="stable")
    used_query = np.zeros(query.mzs.size, dtype=bool)
    used_reference = np.zeros(mzs.size, dtype=bool)
    score = 0.0
    for index in order:
        qi, ri = query_indices[index], reference_indices[index]
        if not used_query[qi] and not used_reference[ri]:
            score += float(products[index])
            used_query[qi] = True
            used_reference[ri] = True
    return min(1.0, score)


def _read_queries(table: pa.Table, max_peaks: int) -> tuple[list[_Query], dict[str, int]]:
    if not isinstance(table, pa.Table):
        raise TypeError("queries must be a pyarrow.Table")
    missing = sorted(set(_QUERY_COLUMNS) - set(table.column_names))
    if missing:
        raise ValueError(f"Missing query columns: {missing}")
    if not table.num_rows:
        raise ValueError("queries must contain at least one spectrum")
    result: list[_Query] = []
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    # Select metadata and peaks explicitly: query structure/label columns are never read.
    for row_number, row in enumerate(table.select(_QUERY_COLUMNS).to_pylist()):
        raw_molecule, raw_spectrum = row["molecule_id"], row["spectrum_id"]
        if raw_molecule is None or raw_spectrum is None:
            raise ValueError(f"Query row {row_number}: null molecule_id or spectrum_id")
        molecule, spectrum = str(raw_molecule), str(raw_spectrum)
        if not molecule.strip() or not spectrum.strip():
            raise ValueError(f"Query row {row_number}: empty molecule_id or spectrum_id")
        if (molecule, spectrum) in seen:
            raise ValueError(f"Duplicate query spectrum: {molecule!r}, {spectrum!r}")
        seen.add((molecule, spectrum))
        adduct = row["adduct"]
        if adduct not in ADDUCT_SHIFTS:
            raise ValueError(f"Query row {row_number}: unsupported adduct {adduct!r}")
        polarity = _polarity(row["ionization_mode"])
        if polarity != (1 if adduct.endswith("+") else -1):
            raise ValueError(f"Query row {row_number}: invalid ionization_mode or adduct polarity mismatch")
        try:
            precursor = float(row["precursor_mz"])
            mass = precursor - ADDUCT_SHIFTS[adduct]
            if not math.isfinite(precursor) or precursor <= 0 or mass <= 0:
                raise ValueError("precursor and neutral mass must be finite and positive")
            mzs, weights = _prepare_peaks(row["ms2_mzs"], row["ms2_normalized_intensities"], max_peaks)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Query row {row_number}: {exc}") from exc
        slot = counts.get(molecule, 0)
        counts[molecule] = slot + 1
        result.append(_Query(molecule, slot, mass, polarity, mzs, weights))
    return result, counts


def search_library(
    train_path: Path,
    queries: pa.Table,
    *,
    ppm: float = 20.0,
    peak_tolerance: float = 0.02,
    top_k: int = 25,
    max_peaks: int = 128,
    batch_size: int = 8192,
    exclude_libraries: set[str] | None = None,
) -> tuple[dict[str, list[str]], dict]:
    """Rank structures by mean best-reference cosine over all molecule spectra.

    Only positive spectral evidence creates a candidate. Unmatched query spectra
    contribute zero to its mean. Score ties use mean neutral-mass error (ppm) of
    the winning positive matches, then inchikey14. Returned lists have at most 25
    entries. Invalid training peaks are counted only inside mass/polarity windows;
    peak arrays outside those windows are deliberately never materialized.
    """
    started = time.perf_counter()
    for name, value in (("ppm", ppm), ("peak_tolerance", peak_tolerance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("top_k", top_k), ("max_peaks", max_peaks), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    excluded = set() if exclude_libraries is None else set(exclude_libraries)
    if any(not isinstance(library, str) for library in excluded):
        raise ValueError("exclude_libraries must contain strings")
    query_list, molecule_sizes = _read_queries(queries, max_peaks)
    library = pq.ParquetFile(train_path)
    missing = sorted(set(_TRAIN_COLUMNS) - set(library.schema_arrow.names))
    if missing:
        raise ValueError(f"Missing training columns: {missing}")
    sorted_queries: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for polarity in (1, -1):
        indices = np.array([i for i, query in enumerate(query_list) if query.polarity == polarity], dtype=np.int64)
        masses = np.array([query_list[i].neutral_mass for i in indices], dtype=np.float64)
        order = np.argsort(masses, kind="stable")
        sorted_queries[polarity] = masses[order], indices[order]
    candidates: dict[str, dict[str, _Candidate]] = {molecule: {} for molecule in molecule_sizes}
    stats = {
        "rows_scanned": 0,
        "excluded_train_rows": 0,
        "unsupported_train_rows": 0,
        "invalid_train_metadata_rows": 0,
        "mass_filtered_train_rows": 0,
        "invalid_filtered_train_spectra": 0,
        "valid_filtered_train_spectra": 0,
        "matched_train_spectra": 0,
        "compared_spectrum_pairs": 0,
        "matched_spectrum_pairs": 0,
        "query_spectra": len(query_list),
        "query_molecules": len(molecule_sizes),
        "parameters": {
            "ppm": float(ppm), "peak_tolerance": float(peak_tolerance),
            "top_k": top_k, "effective_top_k": min(top_k, 25),
            "max_peaks": max_peaks, "batch_size": batch_size,
            "exclude_libraries": sorted(excluded),
        },
        "invalid_peak_count_scope": "mass/polarity-filtered, nonexcluded, supported training rows only",
        "scoring": "sqrt-intensity cosine; greedy one-to-one; mean of best reference per query spectrum; missing=0",
    }
    fraction = ppm * 1e-6
    next_progress = 500_000
    print(f"Retrieving {len(query_list)} spectra for {len(molecule_sizes)} molecules", flush=True)
    for batch in library.iter_batches(batch_size=batch_size, columns=list(_TRAIN_COLUMNS)):
        size = batch.num_rows
        stats["rows_scanned"] += size
        columns = {name: batch.column(batch.schema.get_field_index(name)) for name in _TRAIN_COLUMNS}
        libraries = columns["ingest_lib"].to_pylist()
        eligible = np.fromiter((name not in excluded for name in libraries), dtype=bool, count=size)
        stats["excluded_train_rows"] += int((~eligible).sum())
        adducts = columns["adduct"].to_pylist()
        shifts = np.fromiter((ADDUCT_SHIFTS.get(adduct, np.nan) for adduct in adducts), dtype=np.float64, count=size)
        supported = np.isfinite(shifts)
        stats["unsupported_train_rows"] += int((eligible & ~supported).sum())
        modes = np.fromiter((_polarity(value) for value in columns["ionization_mode"].to_pylist()), dtype=np.int8, count=size)
        expected_modes = np.fromiter((1 if isinstance(adduct, str) and adduct.endswith("+") else -1 for adduct in adducts), dtype=np.int8, count=size)
        precursor = columns["precursor_mz"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        neutral = precursor - shifts
        valid = np.isfinite(precursor) & (precursor > 0) & np.isfinite(neutral) & (neutral > 0) & (modes == expected_modes)
        stats["invalid_train_metadata_rows"] += int((eligible & supported & ~valid).sum())
        valid &= eligible & supported
        lower_indices = np.zeros(size, dtype=np.int64)
        upper_indices = np.zeros(size, dtype=np.int64)
        for polarity, (masses, _) in sorted_queries.items():
            rows = np.flatnonzero(valid & (modes == polarity))
            if not rows.size or not masses.size:
                continue
            # Invert abs(reference-query) <= query*ppm, including very wide ppm.
            lower_indices[rows] = np.searchsorted(masses, neutral[rows] / (1 + fraction), side="left")
            upper_indices[rows] = np.searchsorted(masses, neutral[rows] / (1 - fraction), side="right") if fraction < 1 else masses.size
        selected_indices = np.flatnonzero(lower_indices < upper_indices)
        stats["mass_filtered_train_rows"] += int(selected_indices.size)
        # Arrow take happens BEFORE list-valued peak columns become Python objects.
        selected = batch.take(pa.array(selected_indices, type=pa.int64()))
        for original_index, row in zip(selected_indices, selected.to_pylist()):
            key, smiles = row["inchikey14"], row["normalized_smiles"]
            if not isinstance(key, str) or not key.strip() or not isinstance(smiles, str) or not smiles.strip():
                stats["invalid_filtered_train_spectra"] += 1
                continue
            try:
                mzs, weights = _prepare_peaks(row["ms2_mzs"], row["ms2_normalized_intensities"], max_peaks)
            except (TypeError, ValueError, OverflowError):
                stats["invalid_filtered_train_spectra"] += 1
                continue
            stats["valid_filtered_train_spectra"] += 1
            indices = sorted_queries[int(modes[original_index])][1]
            matched = False
            for query_index in indices[lower_indices[original_index]:upper_indices[original_index]]:
                query = query_list[query_index]
                stats["compared_spectrum_pairs"] += 1
                score = _cosine(query, mzs, weights, peak_tolerance)
                if score <= 0:
                    continue
                matched = True
                stats["matched_spectrum_pairs"] += 1
                error = abs(float(neutral[original_index]) - query.neutral_mass) / query.neutral_mass * 1e6
                molecule_candidates = candidates[query.molecule_id]
                candidate = molecule_candidates.get(key)
                if candidate is None:
                    count = molecule_sizes[query.molecule_id]
                    candidate = _Candidate(smiles, np.zeros(count), np.full(count, np.inf))
                    molecule_candidates[key] = candidate
                elif smiles < candidate.smiles:
                    candidate.smiles = smiles
                slot = query.slot
                if score > candidate.scores[slot] or (score == candidate.scores[slot] and error < candidate.errors[slot]):
                    candidate.scores[slot] = score
                    candidate.errors[slot] = error
            stats["matched_train_spectra"] += int(matched)
        if stats["rows_scanned"] >= next_progress:
            print(f"Scanned {stats['rows_scanned']:,} library rows; matched {stats['matched_train_spectra']:,}", flush=True)
            next_progress += 500_000
    predictions: dict[str, list[str]] = {}
    candidate_counts: dict[str, int] = {}
    for molecule, structures in candidates.items():
        ranked = sorted(
            structures.items(),
            key=lambda item: (
                -float(item[1].scores.mean()),
                float(item[1].errors[np.isfinite(item[1].errors)].mean()),
                item[0],
            ),
        )
        predictions[molecule] = [candidate.smiles for _, candidate in ranked[:min(top_k, 25)]]
        candidate_counts[molecule] = len(ranked)
    no_candidates = sorted(molecule for molecule, values in predictions.items() if not values)
    stats.update({
        "invalid_train_rows": stats["invalid_train_metadata_rows"] + stats["invalid_filtered_train_spectra"],
        "candidate_counts": candidate_counts,
        "covered_molecules": len(predictions) - len(no_candidates),
        "candidate_coverage": (len(predictions) - len(no_candidates)) / len(predictions),
        "no_candidate_ids": no_candidates,
        "elapsed_seconds": time.perf_counter() - started,
    })
    print(f"Retrieved candidates for {stats['covered_molecules']}/{len(predictions)} molecules in {stats['elapsed_seconds']:.1f}s", flush=True)
    return predictions, stats
