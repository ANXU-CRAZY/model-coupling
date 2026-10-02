"""Synthetic file contracts for InVEST preparation; no official HQ execution."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from wetland_coupling.audit import file_sha256
from wetland_coupling.parameters import make_run_plan, validate_prior


class InVESTPlanProvenanceContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = {
            "status": "evidence_reviewed", "target_invest_version": "3.16.1",
            "distance_unit": "m", "fixed_half_saturation_constant": .5,
            "parameters": [{"key": "weight:urban_structure", "low": .5, "high": 1.,
                            "evidence": "synthetic://contract_fixture_only"}],
        }
        self.threats = pd.DataFrame({
            "threat": ["urban_structure"], "max_dist": [1500.], "weight": [1.],
            "decay": ["linear"], "cur_path": ["pressure.tif"],
        })
        self.sensitivity = pd.DataFrame({
            "lucode": [1], "habitat": [1.], "urban_structure": [.5],
        })
        self.write_raster("lulc.tif", np.array([[1, 1], [1, 255]], dtype="uint8"), nodata=255)
        self.write_raster("pressure.tif", np.array([[0, .5], [1., -9999]], dtype="float32"), nodata=-9999)

    def write_raster(self, name, data, *, crs="EPSG:32649", nodata=None):
        with rasterio.open(self.root/name, "w", driver="GTiff", width=2, height=2,
                           count=1, dtype=data.dtype, crs=crs, nodata=nodata,
                           transform=from_origin(657500, 3875600, 100, 100)) as ds:
            ds.write(data, 1)

    def plan(self):
        (self.root/"prior.json").write_text(json.dumps(self.spec), encoding="utf-8")
        self.threats.to_csv(self.root/"threats.csv", index=False)
        self.sensitivity.to_csv(self.root/"sensitivity.csv", index=False)
        return make_run_plan(self.root/"prior.json", self.root/"threats.csv",
                             self.root/"sensitivity.csv", self.root/"lulc.tif",
                             self.root/"plan", members=2, seed=42)

    def assert_no_plan(self):
        self.assertFalse((self.root/"plan").exists())

    def test_sources_and_generated_tables_are_hashed(self):
        plan = self.plan()
        self.assertEqual(plan["status"], "PRIOR_RUN_PLAN_ONLY")
        self.assertEqual(plan["distance_unit"], "m")
        sources = plan["source_inputs"]
        for key, name in (("prior", "prior.json"), ("threats_table", "threats.csv"),
                          ("sensitivity_table", "sensitivity.csv"), ("lulc", "lulc.tif")):
            self.assertEqual(sources[key]["sha256"], file_sha256(self.root/name))
            self.assertEqual(sources[key]["path"], str((self.root/name).resolve()))
        self.assertEqual(sources["lulc"]["raster"]["lucodes"], [1])
        self.assertEqual(sources["lulc"]["raster"]["valid_pixels"], 3)
        self.assertEqual(sources["lulc"]["raster"]["hq_working_pixel_size_m"], 100.)
        pressure = sources["threat_rasters"][0]
        self.assertEqual(pressure["sha256"], file_sha256(self.root/"pressure.tif"))
        self.assertEqual(pressure["threat"], "urban_structure")
        self.assertEqual(pressure["table_column"], "cur_path")
        for member in plan["members"]:
            for record in member["generated_inputs"].values():
                self.assertEqual(record["sha256"], file_sha256(record["path"]))
            table = pd.read_csv(self.root/"plan"/member["member_id"]/"threats.csv")
            self.assertEqual(table.cur_path.iloc[0], pressure["path"])
            self.assertEqual(table.max_dist.iloc[0], 1500.)
        self.assertFalse(list((self.root/"plan").glob("*/official_output")))

    def test_optional_nonempty_threat_paths_are_hashed(self):
        self.write_raster("future_pressure.tif", np.full((2, 2), .25, dtype="float32"))
        self.threats["fut_path"] = "future_pressure.tif"
        self.threats["base_path"] = ""
        records = self.plan()["source_inputs"]["threat_rasters"]
        self.assertEqual({r["table_column"] for r in records}, {"cur_path", "fut_path"})

    def test_missing_current_threat_file_fails_before_output(self):
        (self.root/"pressure.tif").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "input file missing"):
            self.plan()
        self.assert_no_plan()

    def test_missing_lulc_file_fails_before_output(self):
        (self.root/"lulc.tif").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "input file missing"):
            self.plan()
        self.assert_no_plan()

    def test_missing_current_path_fails_before_output(self):
        self.threats["cur_path"] = ""
        with self.assertRaisesRegex(ValueError, "Current threat path missing"):
            self.plan()
        self.assert_no_plan()

    def test_nonraster_threat_file_fails_before_output(self):
        (self.root/"pressure.tif").write_bytes(b"not a raster")
        with self.assertRaisesRegex(ValueError, "Cannot read InVEST raster"):
            self.plan()
        self.assert_no_plan()

    def test_kilometre_declaration_is_rejected(self):
        self.spec["distance_unit"] = "km"
        with self.assertRaisesRegex(ValueError, "Use metres"):
            self.plan()
        self.assert_no_plan()

    def test_maximum_distance_below_pixel_size_is_rejected(self):
        self.threats["max_dist"] = 99.
        with self.assertRaisesRegex(ValueError, "LULC working pixel size"):
            self.plan()
        self.assert_no_plan()

    def test_sampled_distance_below_pixel_size_is_rejected(self):
        self.spec["parameters"].append({"key": "distance:urban_structure", "low": 99.,
                                         "high": 99., "evidence": "synthetic://test_only"})
        with self.assertRaisesRegex(ValueError, "LULC working pixel size"):
            self.plan()
        self.assert_no_plan()

    def test_maximum_distance_equal_to_pixel_size_is_allowed(self):
        self.threats["max_dist"] = 100.
        self.assertEqual(len(self.plan()["members"]), 2)

    def test_lulc_coordinate_units_must_be_metres(self):
        self.write_raster("lulc.tif", np.ones((2, 2), dtype="uint8"), crs="EPSG:2277")
        with self.assertRaisesRegex(ValueError, "linear units must be metres"):
            self.plan()
        self.assert_no_plan()

    def test_threat_values_outside_unit_interval_are_rejected(self):
        self.write_raster("pressure.tif", np.full((2, 2), 1.2, dtype="float32"))
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.plan()
        self.assert_no_plan()

    def test_lulc_codes_require_matching_sensitivity_rows(self):
        self.write_raster("lulc.tif", np.array([[1, 2], [1, 2]], dtype="uint8"))
        with self.assertRaisesRegex(ValueError, "codes missing from sensitivity table"):
            self.plan()
        self.assert_no_plan()

    def test_all_nodata_threat_is_rejected(self):
        self.write_raster("pressure.tif", np.full((2, 2), -9999, dtype="float32"), nodata=-9999)
        with self.assertRaisesRegex(ValueError, "no valid pixels"):
            self.plan()
        self.assert_no_plan()

    def test_pending_project_prior_remains_empty_and_unapproved(self):
        prior = json.loads((Path(__file__).resolve().parents[1]/
                            "configs/invest_prior.pending.json").read_text(encoding="utf-8"))
        self.assertEqual(prior["status"], "pending_evidence")
        self.assertIsNone(prior["fixed_half_saturation_constant"])
        for parameter in prior["parameters"]:
            for field in ("low", "high", "evidence"):
                self.assertIsNone(parameter[field])
        self.assertEqual({p["key"].split(":")[1] for p in prior["parameters"] if ":" in p["key"]},
                         {"urban_structure", "human_activity", "night_light"})
        with self.assertRaisesRegex(ValueError, "not approved"):
            validate_prior(prior)


if __name__ == "__main__":
    unittest.main()
