import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from wetland_coupling.maxent_splits import (
    array_hash, balanced_group_assignment, buffered_group_exclusion,
    build_spatial_split_plan, load_spatial_split_plan, rows_in_mask,
    save_spatial_split_plan, validate_split_plan,
)


class FormalSpatialSplitContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = {"crs": "EPSG:32649", "height": 60, "width": 60,
                    "transform": [100., 0., 650000., 0., -100., 3870000.]}
        cls.valid = np.ones((60, 60), dtype=bool)
        cls.valid[0, 0] = False
        rr, cc = np.indices(cls.valid.shape)
        eligible = cls.valid & (rr % 2 == 0) & (cc % 2 == 0)
        r, c = np.nonzero(eligible)
        cls.table = pd.DataFrame({"native_cell_id": [f"r{a}c{b}" for a, b in zip(r, c)],
            "raster_row": r, "raster_col": c, "x_utm49": 650000. + (c + .5) * 100,
            "y_utm49": 3870000. - (r + .5) * 100, "predictor": r + c})
        cls.presence = {s: cls.table.copy() for s in ("spring", "summer", "autumn", "winter")}
        cls.background = {s: {"B0": cls.table.copy(), "B1": cls.table.copy()} for s in cls.presence}
        cls.config = {"seed": 32, "spatial": {"block_size_m": 1000, "buffer_m": 100,
            "outer_folds": 3, "inner_folds": 3, "locked_test_fraction_target": .2,
            "assignment_seed_trials": 8, "minimum_fit_presences": 20,
            "minimum_validation_presences": 10, "minimum_background": 10}}
        cls.plan = build_spatial_split_plan(cls.presence, cls.background, cls.grid, cls.valid, cls.config)

    def test_locked_groups_and_polygon_buffer_excluded_from_every_fit_and_tune(self):
        p = self.plan; lock = p["masks"]["locked_group_buffer"]
        for outer in p["outer_folds"]:
            self.assertFalse(set(p["locked_groups"]) & set(outer["fit_groups"]))
            self.assertFalse(set(p["locked_groups"]) & set(outer["tune_groups"]))
            for fold in [outer, *outer["inner_folds"]]:
                self.assertFalse(np.any(fold["fit_mask"] & lock))
                self.assertFalse(np.any(fold["validation_mask"] & lock))
        for fold in p["final_inner_folds"]:
            self.assertFalse(np.any((fold["fit_mask"] | fold["validation_mask"]) & lock))

    def test_oof_own_group_never_fits_or_tunes(self):
        for outer in self.plan["outer_folds"]:
            predicted = set(outer["validation_groups"])
            self.assertFalse(predicted & set(outer["fit_groups"]))
            self.assertFalse(predicted & set(outer["tune_groups"]))
            for inner in outer["inner_folds"]:
                self.assertFalse(predicted & set(inner["fit_groups"]))
                self.assertFalse(predicted & set(inner["validation_groups"]))

    def test_variant_and_background_share_identical_cell_groups(self):
        expected = self.plan["presence_by_season"]["spring"][["native_cell_id", "group_id", "outer_fold", "split_role"]]
        for season in self.presence:
            actual = self.plan["presence_by_season"][season]
            pd.testing.assert_frame_equal(expected, actual[expected.columns])
            for b in self.plan["background_by_season"][season].values():
                pd.testing.assert_frame_equal(expected, b[expected.columns])
        # Removing LST/predictors leaves splitting unchanged.
        without = {s: p.drop(columns="predictor") for s, p in self.presence.items()}
        alternate = build_spatial_split_plan(without, self.background, self.grid, self.valid, self.config)
        self.assertEqual(self.plan["split_hash"], alternate["split_hash"])

    def test_unsurveyed_and_nodata_grid_cells_also_have_groups_and_roles(self):
        self.assertEqual(self.plan["group_raster"].shape, self.valid.shape)
        self.assertTrue((self.plan["group_raster"] > 0).all())
        self.assertEqual(self.plan["masks"]["role_raster"].shape, self.valid.shape)
        self.assertFalse(self.plan["development_fit_mask"][0, 0])

    def test_actual_fit_validation_cells_are_more_than_buffer_apart(self):
        for outer in self.plan["outer_folds"]:
            fit = np.argwhere(outer["fit_mask"]); val = np.argwhere(outer["validation_mask"])
            distance = cKDTree(val * 100).query(fit * 100)[0].min()
            self.assertGreater(distance, self.config["spatial"]["buffer_m"])

    def test_site_identity_merges_blocks_before_role_assignment(self):
        sites = pd.DataFrame({"site_key": ["same_site", "same_site"], "raster_row": [4, 4], "raster_col": [8, 10]})
        merged = build_spatial_split_plan(self.presence, self.background, self.grid, self.valid, self.config, sites)
        self.assertEqual(merged["group_raster"][4, 8], merged["group_raster"][4, 10])
        self.assertEqual(merged["named_site_block_merges"], 1)

    def test_deterministic_rerun_and_saved_hash_verification(self):
        repeated = build_spatial_split_plan(self.presence, self.background, self.grid, self.valid, self.config)
        self.assertEqual(self.plan["split_hash"], repeated["split_hash"])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "split"
            save_spatial_split_plan(self.plan, out, self.config)
            loaded = load_spatial_split_plan(out)
            self.assertEqual(loaded["split_hash"], self.plan["split_hash"])
            with (out / "config_snapshot.json").open("a") as stream: stream.write(" ")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                load_spatial_split_plan(out)

    def test_mutated_mask_and_leaking_provenance_fail_closed(self):
        bad = copy.deepcopy(self.plan)
        bad["masks"]["development_fit"][0, 0] = True
        with self.assertRaises(ValueError): validate_split_plan(bad)
        bad = copy.deepcopy(self.plan)
        bad["outer_folds"][0]["tune_groups"].append(bad["locked_groups"][0])
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_split_plan(bad)

    def test_missing_crs_nodata_and_position_mismatch_fail_closed(self):
        grid = dict(self.grid); grid["crs"] = "EPSG:3857"
        with self.assertRaisesRegex(ValueError, "32649"):
            build_spatial_split_plan(self.presence, self.background, grid, self.valid, self.config)
        bad = {s: p.copy() for s, p in self.presence.items()}
        bad["spring"].loc[0, "x_utm49"] += 100
        with self.assertRaisesRegex(ValueError, "centers"):
            build_spatial_split_plan(bad, self.background, self.grid, self.valid, self.config)
        valid = self.valid.copy(); first = self.table.iloc[0]; valid[int(first.raster_row), int(first.raster_col)] = False
        with self.assertRaisesRegex(ValueError, "NoData"):
            build_spatial_split_plan(self.presence, self.background, self.grid, valid, self.config)

    def test_background_remains_presence_only_candidate(self):
        self.assertFalse(self.plan["background_is_absence"])
        for backgrounds in self.plan["background_by_season"].values():
            for b in backgrounds.values(): self.assertNotIn("absence", b.columns)
        self.assertTrue(self.plan["lst_variants_share_all_spatial_partitions"])
        bad = copy.deepcopy(self.background)
        bad["spring"]["B0"]["absence"] = True
        with self.assertRaisesRegex(ValueError, "absence"):
            build_spatial_split_plan(self.presence, bad, self.grid, self.valid, self.config)

    def test_runtime_mask_alias_cannot_bypass_frozen_masks(self):
        bad = copy.deepcopy(self.plan)
        bad["outer_folds"][0]["fit_mask"] = self.valid.copy()
        with self.assertRaisesRegex(ValueError, "Runtime"):
            validate_split_plan(bad)

    def test_buffer_is_defined_from_unsurveyed_heldout_polygon(self):
        groups = np.ones((40, 40), dtype=int); groups[:, 20:] = 2
        held = buffered_group_exclusion(groups, [2], 500)
        self.assertTrue(held[:, 19].all())
        self.assertTrue(held[:, 15].all())
        self.assertFalse(held[:, 10].any())

    def test_low_count_does_not_silently_relax_split(self):
        scarce = {s: p.iloc[:10].copy() for s, p in self.presence.items()}
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            build_spatial_split_plan(scarce, self.background, self.grid, self.valid, self.config)


if __name__ == "__main__":
    unittest.main()
