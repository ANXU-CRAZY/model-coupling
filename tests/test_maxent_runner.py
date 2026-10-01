"""Runner orchestration contracts using artificial grids and mocked model calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from wetland_coupling.maxent_inputs import dataframe_cells
from wetland_coupling.maxent_protocol import freeze_selection, claim_locked_test


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "zhengzhou" / "run_maxent_nested_cv.py"
SPEC = importlib.util.spec_from_file_location("maxent_runner_contract_fixture", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture():
    grid = {"height": 6, "width": 6, "crs": "EPSG:32649", "epsg": 32649,
            "resolution_m": 100, "transform": [100., 0., 650000., 0., -100., 3800000.]}
    names = ["cov_a", "cov_b", "lst_reference"]
    idx = np.arange(36).reshape(6, 6)
    env = np.stack([idx / 36., np.sin(idx * 1.37), (idx % 6) / 5.], axis=-1)
    cells = dataframe_cells(np.arange(36), env, names, grid)
    visits = cells.iloc[:, :7].copy()
    visits["inferred_visit_count"] = 1 + np.arange(36) % 3
    ctx = {"grid": grid, "env": env, "names": names, "valid_mask": np.ones((6, 6), dtype=bool),
           "presence": cells.copy(), "reference": cells.copy(), "B0": cells.copy(), "visits": visits}
    fit = np.zeros((6, 6), dtype=bool)
    fit[:3] = True
    val = ~fit
    config = {"seed": 71,
              "spatial": {"minimum_fit_presences": 2, "minimum_validation_presences": 2, "minimum_background": 2},
              "background": {"is_absence": False, "uniform_points_per_fit": 8,
                             "kde_bandwidth_m": 100., "uniform_mixture_fraction": .1},
              "predictors": {"absolute_spearman_cutoff": .95},
              "maxent": {"rm": [.5, 1.], "fc": ["L", "LQ"], "max_iterations": 3, "parallel_workers": 1},
              "selection": {"mean_omission_eligibility_max": .2}}
    return ctx, fit, val, config


class MaxentRunnerContracts(unittest.TestCase):
    def test_final_feature_plan_uses_only_fit_domain_for_both_background_types(self):
        for scheme in ("B1_uniform", "B2_visit_density_proxy"):
            ctx, fit, val, config = fixture()
            before = runner.final_feature_plan(ctx, fit, scheme, "static_lst", config, 19)
            ctx["env"][val] = 987654.
            ctx["presence"].loc[18:, "cov_a"] = -777.
            ctx["reference"].loc[18:, "cov_b"] = 999.
            ctx["visits"].loc[18:, "inferred_visit_count"] = -1
            after = runner.final_feature_plan(ctx, fit, scheme, "static_lst", config, 19)
            self.assertEqual(before, after)
            self.assertFalse(before["background_audit"]["validation_or_locked_visits_used_for_bias"])

    def test_final_features_are_deterministic_and_lst_variants_share_background_rows(self):
        ctx, fit, _, config = fixture()
        no_lst = runner.final_feature_plan(ctx, fit, "B1_uniform", "no_lst", config, 23)
        static = runner.final_feature_plan(ctx, fit, "B1_uniform", "static_lst", config, 23)
        self.assertEqual(no_lst, runner.final_feature_plan(ctx, fit, "B1_uniform", "no_lst", config, 23))
        self.assertFalse(any(name.startswith("lst_") for name in no_lst["predictors"]))
        self.assertIn("lst_reference", static["predictors"])
        self.assertEqual(no_lst["background_frame_sha256"], static["background_frame_sha256"])
        self.assertEqual(no_lst["background_audit"], static["background_audit"])

    def test_prepare_fit_uses_same_spatial_rows_and_append_slice_for_both_variants(self):
        ctx, fit, val, config = fixture()
        with tempfile.TemporaryDirectory() as folder:
            outputs = []
            for variant in runner.VARIANTS:
                scope = Path(folder) / variant
                meta = runner.prepare_fit(scope, ctx, fit, val, "B1_uniform", variant, config, 23, True)
                frozen = runner.final_feature_plan(ctx, fit, "B1_uniform", variant, config, 23)
                self.assertEqual(meta["predictors"], frozen["predictors"])
                table = pd.read_csv(scope / "projection.csv")
                start = meta["projection_start"]
                self.assertEqual(start, meta["offsets"][-1])
                self.assertEqual(len(table) - start, len(meta["raster_indices"]))
                np.testing.assert_array_equal(meta["raster_indices"], np.flatnonzero(val))
                expected = dataframe_cells(meta["raster_indices"], ctx["env"], ctx["names"], ctx["grid"])
                np.testing.assert_allclose(table.iloc[start:, 1:3], expected[["longitude", "latitude"]], atol=1e-9, rtol=0)
                outputs.append((meta, pd.read_csv(scope / "train.csv"), pd.read_csv(scope / "background.csv")))
            self.assertEqual(outputs[0][0]["offsets"], outputs[1][0]["offsets"])
            for frame_index in (1, 2):
                np.testing.assert_array_equal(outputs[0][frame_index].iloc[:, :3], outputs[1][frame_index].iloc[:, :3])

    def test_result_metrics_excludes_appended_map_rows_from_validation(self):
        values = np.array([.7, .8, .1, .2, .6, .9, .15, .25, 123., -10.])
        meta = {"offsets": [0, 2, 4, 6, 8], "projection_start": 8}
        with patch.object(runner, "metrics", return_value={"fixture": True}) as measured:
            self.assertEqual(runner.result_metrics({"predictions": values, "complexity": 4}, meta), {"fixture": True})
        args = measured.call_args.args
        for actual, expected in zip(args[:4], (values[:2], values[2:4], values[4:6], values[6:8])):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(args[4], 4)

    def test_tuning_pairs_fold_masks_and_sampling_seed_and_batches_complete_grid(self):
        ctx, fit, val, config = fixture()
        folds = [{"fit_mask": fit, "validation_mask": val, "fit_groups": [1], "validation_groups": [2]},
                 {"fit_mask": val, "validation_mask": fit, "fit_groups": [2], "validation_groups": [1]}]
        calls, batch_sizes = [], []
        def prepared(scope, context, fit_mask, val_mask, background, variant, config_value, seed):
            calls.append((fit_mask, val_mask, background, variant, seed))
            return {"predictors": ["cov_a"], "offsets": [0, 2, 4, 6, 8]}
        def model_jobs(jobs, max_workers):
            batch_sizes.append((len(jobs), max_workers))
            return [{"predictions": np.array([.7, .8, .1, .2, .6, .9, .15, .25]),
                     "complexity": 2, "manifest_path": str(job["out_dir"] / "manifest.json")} for job in jobs]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "reports").mkdir()
            with patch.object(runner, "prepare_fit", side_effect=prepared), patch.object(runner, "run_jobs", side_effect=model_jobs):
                runner.tune_scope(root, "spring", "artificial_scope", ctx, folds, config,
                                  SimpleNamespace(java="unused", jar="unused"))
        self.assertTrue(batch_sizes)
        self.assertTrue(all(value == (4, 1) for value in batch_sizes))
        for offset in range(0, len(calls), 2):
            a, b = calls[offset:offset + 2]
            self.assertIs(a[0], b[0])
            self.assertIs(a[1], b[1])
            self.assertEqual(a[2], b[2])
            self.assertEqual((a[3], b[3]), runner.VARIANTS)
            self.assertEqual(a[4], b[4])

    def test_frozen_actual_features_reject_edit_before_exclusive_test_claim(self):
        ctx, fit, _, config = fixture()
        selection = {"spring": {"final_features": {
            "no_lst": runner.final_feature_plan(ctx, fit, "B1_uniform", "no_lst", config, 23)}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frozen.json"
            frozen_hash = freeze_selection(path, selection, "split_fixture", "config_fixture")
            stored = json.loads(path.read_text(encoding="utf-8"))
            stored["selection"]["spring"]["final_features"]["no_lst"]["predictors"] = ["changed"]
            path.write_text(json.dumps(stored), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash/status changed"):
                claim_locked_test(folder, path, frozen_hash, "split_fixture", "config_fixture")
            self.assertFalse((Path(folder) / "LOCKED_TEST_ATTEMPT.json").exists())


if __name__ == "__main__":
    unittest.main()
