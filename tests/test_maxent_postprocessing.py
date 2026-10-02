import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.zhengzhou.evaluate_maxent_cv import paired_outer_lst_metrics, projection_member_oof, primary_pipeline_scope, postprocessing_provenance, selection_constraint_audit
from scripts.zhengzhou.export_maxent_oof import (
    validate_member,
    validate_oof_arrays,
    verify_split_inventory,
)


class ConditionalOofContracts(unittest.TestCase):
    def test_declared_empty_validation_group_is_distinct_from_predicted_groups(self):
        provenance = {
            "strict_end_to_end_oof": False,
            "gate_eligible": False,
            "fit_groups": [1],
            "tune_groups": [1],
            "validation_groups": [2, 3],
            "calibrate_groups": [],
            "output_scale": "cloglog",
            "background_is_absence": False,
        }
        fold = {"fit_groups": [1], "validation_groups": [2, 3]}
        validate_member(provenance, fold, [9], [2])

    def test_member_provenance_must_match_declared_groups(self):
        provenance = {
            "strict_end_to_end_oof": False,
            "gate_eligible": False,
            "fit_groups": [1],
            "tune_groups": [1],
            "validation_groups": [2],
            "output_scale": "cloglog",
            "background_is_absence": False,
        }
        with self.assertRaisesRegex(ValueError, "declared validation groups"):
            validate_member(provenance, {"fit_groups": [1], "validation_groups": [2, 3]}, [9], [2])

    def test_oof_fold_mask_and_member_count_must_match_frozen_domain(self):
        masks = {
            "common_valid": np.ones((1, 4), dtype=bool),
            "locked_test": np.zeros((1, 4), dtype=bool),
            "group_raster": np.array([[10, 10, 20, 20]]),
            "outer_0_validation": np.array([[1, 1, 0, 0]], dtype=bool),
            "outer_1_validation": np.array([[0, 0, 1, 1]], dtype=bool),
        }
        plan = {"outer_folds": [{"fold": 0}, {"fold": 1}]}
        arrays = {
            "M_oof": np.array([.2, .3, .4, .5]),
            "q05": np.array([.1, .2, .3, .4]),
            "median": np.array([.2, .3, .4, .5]),
            "q95": np.array([.3, .4, .5, .6]),
            "std": np.array([.01, .02, .03, .04]),
            "member_count": np.array([3, 3, 3, 3]),
            "fold": np.array([0, 0, 1, 1]),
        }
        np.testing.assert_array_equal(validate_oof_arrays(arrays, masks, plan), [0, 0, 1, 1])
        arrays["fold"] = np.array([0, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "OOF fold does not match"):
            validate_oof_arrays(arrays, masks, plan)

    def test_split_inventory_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = {}
            for name in ("split_plan.json", "split_masks.npz", "config_snapshot.json"):
                path = root / name
                path.write_text(name, encoding="utf-8")
                records[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (root / "manifest.json").write_text(json.dumps({"outputs": records}), encoding="utf-8")
            verify_split_inventory(root)
            (root / "split_masks.npz").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_split_inventory(root)


class PairedOuterMetricContracts(unittest.TestCase):
    def test_postprocessing_source_does_not_claim_training_commit(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        with patch("scripts.zhengzhou.evaluate_maxent_cv.subprocess.run", side_effect=[
                SimpleNamespace(stdout="current-head\n"), SimpleNamespace(stdout=" M scripts/zhengzhou/evaluate_maxent_cv.py\n")]):
            result = postprocessing_provenance("training-head")
        self.assertEqual(result["training_git_commit"], "training-head")
        self.assertEqual(result["postprocessing_git_commit"], "current-head")
        self.assertTrue(result["dirty_worktree"])
        self.assertNotIn("git_commit", result)
        self.assertEqual(set(result["source_code_sha256"]), {
            "scripts/zhengzhou/evaluate_maxent_cv.py", "scripts/zhengzhou/export_maxent_oof.py"})

    def test_lst_comparison_uses_same_b1_background_not_variant_primary(self):
        rows = []
        for fold in (0, 1):
            for variant in ("no_lst", "static_lst"):
                for background, auc in (("B0_target_group", .95), ("B1_uniform", .70), ("B2_visit_density_proxy", .60)):
                    if variant == "static_lst":
                        auc += .10
                    rows.append({"season": "spring", "fold": fold, "variant": variant,
                                 "background": background, "validation_auc": auc,
                                 "chosen_background": (background == ("B0_target_group" if variant == "no_lst" else "B2_visit_density_proxy"))})
        measures = ["validation_auc"]
        result = paired_outer_lst_metrics(pd.DataFrame(rows), measures)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(row["reference_background"] == "B1_uniform" for row in result))
        self.assertTrue(all(row["metric_scope"].startswith("B1-specific") for row in result))
        self.assertAlmostEqual(result[0]["validation_auc_static_minus_no_lst"], .10)

    def test_missing_b1_fold_fails_closed(self):
        rows = []
        for fold in (0, 1):
            rows.append({"season": "spring", "fold": fold, "variant": "no_lst",
                         "background": "B1_uniform", "validation_auc": .7})
        rows.append({"season": "spring", "fold": 0, "variant": "static_lst",
                     "background": "B1_uniform", "validation_auc": .8})
        with self.assertRaisesRegex(ValueError, "Incomplete paired B1-specific"):
            paired_outer_lst_metrics(pd.DataFrame(rows), ["validation_auc"])

    def test_primary_pipeline_scope_exposes_fitted_background_mismatch(self):
        rows = []
        for fold in (0, 1):
            for variant in ("no_lst", "static_lst"):
                background = "B2_visit_density_proxy" if variant == "static_lst" and fold == 1 else "B1_uniform"
                rows.append({"season": "spring", "variant": variant, "fold": fold,
                             "background": background, "chosen_background": True})
        scope = primary_pipeline_scope(pd.DataFrame(rows), "spring")
        self.assertTrue(scope["fitted_backgrounds_differ"])
        self.assertFalse(scope["isolated_lst_comparison"])
        self.assertEqual(scope["background_mismatch_folds"], "1")


class PreservedProjectionOutputContracts(unittest.TestCase):
    def setUp(self):
        from rasterio.warp import transform
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine_path = self.root / "manifest.json"
        self.selection_path = self.root / "selection.json"
        self.grid = {"height": 2, "width": 2, "transform_gdal": [500000, 100, 0, 3900000, 0, -100]}
        self.indices = np.array([0, 2, 3])
        lon, lat = transform("EPSG:32649", "EPSG:4326", [500050, 500050, 500150], [3899950, 3899850, 3899850])
        tail = pd.DataFrame({"longitude": lon, "latitude": lat, "bird cloglog values": [.2, .4, .7]})
        prefix = pd.concat([tail.iloc[[0]]] * 4, ignore_index=True)
        full = pd.concat([prefix, tail], ignore_index=True)
        self.parts = [self.root / f"bird_projection_part{i:05d}.csv" for i in range(2)]
        full.iloc[:5].to_csv(self.parts[0], index=False)
        full.iloc[5:].to_csv(self.parts[1], index=False)
        self.engine = {"status": "OFFICIAL_MAXENT_FITTED", "output_scale": "cloglog",
                       "output_is_calibrated_probability": False, "background_is_absence": False,
                       "prediction_rows": 7, "inputs": {"train": {"rows": 1, "columns": ["species", "longitude", "latitude", "dem"]},
                       "projection": {"rows": 7}}, "outputs": {}}
        self.selection = {"predictors": ["dem"], "offsets": [0, 1, 2, 3, 4], "projection_start": 4}
        self.refresh()

    def tearDown(self):
        self.temp.cleanup()

    def refresh(self):
        self.engine["outputs"] = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                                  for p in self.parts}
        self.engine_path.write_text(json.dumps(self.engine), encoding="utf-8")
        self.selection_path.write_text(json.dumps(self.selection), encoding="utf-8")
        self.engine_hash = hashlib.sha256(self.engine_path.read_bytes()).hexdigest()

    def recover(self):
        return projection_member_oof(self.engine_path, self.engine_hash, self.selection_path, self.indices, self.grid)

    def test_recover_tail_across_parts_without_deleted_projection_inputs(self):
        values, lineage, consumed = self.recover()
        np.testing.assert_allclose(values, [.2, .4, .7])
        self.assertEqual(lineage["oof_pixels"], 3)
        self.assertTrue(lineage["every_output_coordinate_verified"])
        self.assertFalse(lineage["deleted_projection_inputs_read"])
        self.assertEqual(set(consumed), {self.engine_path, self.selection_path, *self.parts})

    def test_output_hash_mismatch_fails_closed(self):
        self.parts[1].write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.recover()

    def test_engine_manifest_hash_mismatch_fails_closed(self):
        self.engine_path.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.recover()

    def test_reordered_output_cells_fail_even_when_inventory_hash_matches(self):
        frame = pd.read_csv(self.parts[1])
        frame.iloc[::-1].to_csv(self.parts[1], index=False)
        self.refresh()
        with self.assertRaisesRegex(ValueError, "row-major OOF cells"):
            self.recover()

    def test_truncated_output_fails_even_when_inventory_hash_matches(self):
        frame = pd.read_csv(self.parts[1])
        frame.iloc[:1].to_csv(self.parts[1], index=False)
        self.refresh()
        with self.assertRaisesRegex(ValueError, "original row count"):
            self.recover()

    def test_wrong_projection_offset_fails_closed(self):
        self.selection["projection_start"] = 3
        self.refresh()
        with self.assertRaisesRegex(ValueError, "tail/offset"):
            self.recover()

    def test_missing_projection_part_fails_closed(self):
        self.engine["outputs"].pop(self.parts[0].name)
        self.engine_path.write_text(json.dumps(self.engine), encoding="utf-8")
        self.engine_hash = hashlib.sha256(self.engine_path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "output part"):
            self.recover()

    def test_invalid_cloglog_score_fails_closed(self):
        frame = pd.read_csv(self.parts[1])
        frame.iloc[-1, 2] = 1.2
        frame.to_csv(self.parts[1], index=False)
        self.refresh()
        with self.assertRaisesRegex(ValueError, "outside"):
            self.recover()


class SelectionQualificationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        (self.run / "reports").mkdir()
        (self.run / "manifests").mkdir()
        self.plan = {"outer_folds": [{"fold": i} for i in range(3)]}
        self.state = {"outputs": {}}
        self.convergence = []
        frozen = {"selection": {}}
        for season in ("spring", "summer", "autumn", "winter"):
            for scope in ("outer_0", "outer_1", "outer_2", "full_development"):
                winners = {}
                for variant in ("no_lst", "static_lst"):
                    failed = season == "spring" and scope == "outer_0" and variant == "no_lst"
                    winners[variant] = {"background": "B1_uniform", "rm": 4., "fc": "L", "inner_folds": 3,
                        "mean_omission": .25 if failed else .1, "omission_constraint_failed": failed}
                    self.convergence.extend({"season": season, "scope": scope, "variant": variant,
                        "background": "B1_uniform", "rm": 4., "fc": "L", "category": "inner_candidate",
                        "official_html_termination": "converged"} for _ in range(3))
                path = self.run / "reports" / f"tuning_{season}_{scope}.json"
                path.write_text(json.dumps({"variant_winners": winners}), encoding="utf-8")
                self.state["outputs"][path.relative_to(self.run).as_posix()] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                if scope == "full_development":
                    frozen["selection"][season] = {"variant_winners": winners}
        path = self.run / "manifests/frozen_selection.json"
        path.write_text(json.dumps(frozen), encoding="utf-8")
        self.state["frozen_selection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def test_report_fallbacks_and_selected_inner_convergence_without_reranking(self):
        rows, summary, consumed = selection_constraint_audit(self.run, self.plan, self.state, self.convergence, .2)
        self.assertEqual(len(rows), 32)
        self.assertEqual(summary["outer_primary_winners"], 24)
        self.assertEqual(summary["outer_primary_fallbacks"], 1)
        self.assertEqual(summary["frozen_final_fallbacks"], 0)
        self.assertEqual(summary["selected_candidate_inner_models_converged"], 96)
        self.assertFalse(summary["selection_rule_modified"])
        self.assertEqual(len(consumed), 17)

    def test_missing_inner_convergence_evidence_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "inner convergence evidence is incomplete"):
            selection_constraint_audit(self.run, self.plan, self.state, self.convergence[1:], .2)

    def test_qualification_flag_must_match_frozen_limit(self):
        with self.assertRaisesRegex(ValueError, "qualification flag is inconsistent"):
            selection_constraint_audit(self.run, self.plan, self.state, self.convergence, .3)


if __name__ == "__main__":
    unittest.main()
