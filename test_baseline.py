"""Consumer-visible chemical identity, ranking and holdout regression cases."""

import csv
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation import (
    evaluate_predictions,
    structure_key,
    validate_submission,
    write_structure_folds,
    write_submission,
)
from retrieval import ADDUCT_SHIFTS, search_library


class ChemicalIdentityTests(unittest.TestCase):
    def test_metric_tautomers_stereochemistry_invalid_rank_and_cutoff(self):
        predictions = {
            "tautomer": ["CC", "invalid_smiles", "Oc1ccccn1"],
            "stereo": ["C[C@@H](O)C(=O)O"],
            "outside_cutoff": ["CC"] * 25 + ["CCO"],
        }
        truth = {
            "tautomer": "O=c1cccc[nH]1",
            "stereo": "C[C@H](O)C(=O)O",
            "outside_cutoff": "CCO",
            "missing": "CCN",
        }
        metrics = evaluate_predictions(predictions, truth)
        self.assertAlmostEqual(metrics["mrr_at_25"], (1 / 3 + 1) / 4)
        self.assertEqual(metrics["top1_accuracy"], 0.25)
        self.assertEqual(metrics["top25_recall"], 0.5)

    def test_submission_deduplicates_chemical_identity_and_rejects_missing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.csv"
            write_submission(
                {"m1": ["Oc1ccccn1", "O=c1cccc[nH]1", "CCO"]}, ["m1"], output
            )
            with output.open(newline="", encoding="utf-8") as handle:
                candidates = next(csv.DictReader(handle))["smiles"].split(";")
            self.assertEqual(candidates, ["Oc1ccccn1", "CCO"])
            self.assertEqual(validate_submission(output, ["m1"])["candidates"], 2)
            with self.assertRaises(ValueError):
                validate_submission(output, ["m1", "m2"])

    def test_tautomer_variants_cannot_cross_structure_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.parquet"
            output = Path(directory) / "folds.parquet"
            pq.write_table(pa.table({
                "inchikey14": ["A" * 14, "B" * 14, "C" * 14],
                "normalized_smiles": ["Oc1ccccn1", "O=c1cccc[nH]1", "CCO"],
            }), source)
            summary = write_structure_folds(source, output)
            rows = {row["inchikey14"]: row for row in pq.read_table(output).to_pylist()}
            self.assertEqual(rows["A" * 14]["split_group"], rows["B" * 14]["split_group"])
            self.assertEqual(rows["A" * 14]["fold"], rows["B" * 14]["fold"])
            self.assertEqual(summary["canonical_structures"], 2)

    def test_conflicting_canonical_keys_stay_in_one_transitive_split_group(self):
        # Two real training representations share this original InChIKey14.
        first = "C=CC1=C(C)C(=Cc2[nH]c(Cc3[nH]c(C=C4NC(=O)C(C)=C4C=C)c(C)c3CCC(=O)O)c(CCC(=O)O)c2C)NC1=O"
        second = "C=CC1=C(C)C(O)=NC1=Cc1[nH]c(Cc2[nH]c(C=C3N=C(O)C(C=C)=C3C)c(C)c2CCC(=O)O)c(CCC(=O)O)c1C"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.parquet"
            output = Path(directory) / "folds.parquet"
            records = [
                {"inchikey14": "BPYKTIZUTYGOLE", "normalized_smiles": first},
                {"inchikey14": "BPYKTIZUTYGOLE", "normalized_smiles": second},
                {"inchikey14": "X" * 14, "normalized_smiles": second},
            ]
            pq.write_table(pa.Table.from_pylist(records), source)
            summary = write_structure_folds(source, output)
            rows = {row["inchikey14"]: row for row in pq.read_table(output).to_pylist()}
            self.assertEqual(summary["ambiguous_raw_structures"], 1)
            self.assertEqual(summary["split_groups"], 1)
            self.assertEqual(set(rows["BPYKTIZUTYGOLE"]["structure_keys"]), {structure_key(first), structure_key(second)})
            self.assertEqual(rows["BPYKTIZUTYGOLE"]["split_group"], rows["X" * 14]["split_group"])
            self.assertEqual(rows["BPYKTIZUTYGOLE"]["fold"], rows["X" * 14]["fold"])
            pq.write_table(pa.Table.from_pylist(list(reversed(records))), source)
            write_structure_folds(source, output)
            reordered = {row["inchikey14"]: row for row in pq.read_table(output).to_pylist()}
            self.assertEqual(rows, reordered)


class SpectralRetrievalTests(unittest.TestCase):
    def test_adduct_mass_polarity_peak_ranking_and_library_holdout(self):
        neutral_mass = 46.041864812

        def reference(smiles, adduct, mode, peaks, intensities, library="reference"):
            return {
                "normalized_smiles": smiles,
                "inchikey14": structure_key(smiles),
                "ingest_lib": library,
                "adduct": adduct,
                "ionization_mode": mode,
                "precursor_mz": neutral_mass + ADDUCT_SHIFTS[adduct],
                "precursor_error_ppm": 0.0,
                "ms2_mzs": peaks,
                "ms2_normalized_intensities": intensities,
                "num_peaks": len(peaks),
            }

        references = [
            reference("CCO", "[M+H]+", "positive", [5., 10.], [1., 0.5]),
            reference("COC", "[M+H]+", "positive", [5., 15.], [1., 1.]),
            reference("COC", "[M-H]-", "negative", [7., 12.], [1., 1.]),
            reference("CCO", "[M-H]-", "negative", [7., 17.], [1., 1.]),
            reference("COC", "[M+H]+", "positive", [5., 10.], [1., 1.], "holdout"),
        ]
        # These independently specified ion masses also catch incorrect adduct offsets.
        queries = pa.Table.from_pylist([
            {"molecule_id": "positive_query", "spectrum_id": "p",
             "adduct": "[M-H2O+H]+", "ionization_mode": "positive",
             "precursor_mz": neutral_mass + 1.007276466621 - 18.01056468403,
             "ms2_mzs": [5., 10.], "ms2_normalized_intensities": [1., 1.]},
            {"molecule_id": "negative_query", "spectrum_id": "n",
             "adduct": "[M+CH2O2-H]-", "ionization_mode": "negative",
             "precursor_mz": neutral_mass + 44.998202837,
             "ms2_mzs": [7., 12.], "ms2_normalized_intensities": [1., 1.]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.parquet"
            pq.write_table(pa.Table.from_pylist(references), source)
            predictions, _ = search_library(source, queries, exclude_libraries={"holdout"})
            self.assertEqual(predictions["positive_query"][0], "CCO")
            self.assertEqual(predictions["negative_query"][0], "COC")
            contaminated, _ = search_library(source, queries)
            self.assertEqual(contaminated["positive_query"][0], "COC")


if __name__ == "__main__":
    unittest.main()
