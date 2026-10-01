import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.zhengzhou.evaluate_maxent_cv import paired_outer_lst_metrics
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


if __name__ == "__main__":
    unittest.main()
