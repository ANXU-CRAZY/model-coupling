"""MaxEnt adapter contracts; synthetic fixture runs never use observation data."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from wetland_coupling.maxent_engine import (
    FC_FLAGS, _worker, inspect_swd, lambda_complexity, run_job, run_jobs, validate_scale,
)


class MaxentEngineContracts(unittest.TestCase):
    def test_cloglog_scale_rejects_bad_values(self):
        for values in ([np.nan], [np.inf], [-0.001], [1.001]):
            with self.assertRaises(ValueError):
                validate_scale(values)
        np.testing.assert_array_equal(validate_scale([0, .5, 1]), [0, .5, 1])

    def test_complexity_excludes_normalizers_and_zero_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.lambdas"
            path.write_text("x, 0, 0, 1\nx^2, 1.5, 0, 1\n'x, -0.7, 0, 1\n"
                            "linearPredictorNormalizer, 4\ndensityNormalizer, 3\n"
                            "numBackgroundPoints, 100\nentropy, 2\n")
            self.assertEqual(lambda_complexity(path), 2)

    def test_swd_missing_predictor_nodata_and_nonfinite_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            for value in (-9999, np.nan, np.inf):
                pd.DataFrame({"species": ["fixture"], "longitude": [113], "latitude": [35], "x": [value]}).to_csv(path, index=False)
                with self.assertRaises((ValueError, TypeError)):
                    inspect_swd(path)
            path.write_text("species,longitude,latitude,x\nfixture,113,35,.5\n")
            with self.assertRaises(ValueError):
                inspect_swd(path, ["species", "longitude", "latitude", "missing"])

    def test_duplicate_headers_rejected_before_pandas_mangles(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            path.write_text("species,longitude,latitude,x,x\nfixture,113,35,.5,.5\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                inspect_swd(path)

    def test_fc_grid_is_explicit_and_existing_output_cannot_be_overwritten(self):
        self.assertEqual(FC_FLAGS, {"L": (True, False, False), "LQ": (True, True, False), "LQH": (True, True, True)})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run_jobs([{"out_dir": folder}], 1)

    def test_duplicate_job_output_refused_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            output = str(Path(folder) / "same")
            with self.assertRaises(FileExistsError):
                run_jobs([{"out_dir": output}, {"out_dir": output}], 1)

    def test_helper_compilation_failure_is_recorded_without_false_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            job = {"out_dir": folder, "java": str(Path(folder) / "java.exe"),
                   "jar": str(Path(folder) / "maxent.jar"), "args": ["visible=false"],
                   "manifest": {"status": "PREPARED_NOT_FITTED"}}
            failed = SimpleNamespace(returncode=1, stdout="", stderr="synthetic compilation failure")
            with patch("wetland_coupling.maxent_engine.subprocess.run", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "compilation failed"):
                    _worker([(0, job)])
            manifest = json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "FAILED_OR_NOT_COMPLETED")
            self.assertIn("synthetic compilation failure",
                          (Path(folder) / "_java_batch" / "compile_stderr.txt").read_text())

    @unittest.skipUnless(os.environ.get("MAXENT_JAR") and os.environ.get("MAXENT_JAVA"),
                         "Set MAXENT_JAR/MAXENT_JAVA to run official synthetic integration")
    def test_official_synthetic_batch_isolation_chunk_order_and_repeatability(self):
        java, jar = os.environ["MAXENT_JAVA"], os.environ["MAXENT_JAR"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            x = np.linspace(.01, .99, 45)
            data = pd.DataFrame({"species": "synthetic_fixture_only", "longitude": 113 + x / 10,
                                 "latitude": 34 + x / 10, "cov_a": x, "cov_b": np.sin(x * 8)})
            train = data.iloc[25:40].copy()
            train.to_csv(root / "train.csv", index=False)
            background = data.copy()
            background["species"] = "background"
            background.to_csv(root / "background.csv", index=False)
            data.to_csv(root / "projection.csv", index=False)
            jobs = [dict(java=java, jar=jar, train_csv=root / "train.csv", background_csv=root / "background.csv",
                         projection_csv=root / "projection.csv", out_dir=root / name, rm=1, fc=fc, seed=42,
                         max_iterations=100, timeout=60) for name, fc in (("lq_a", "LQ"), ("l", "L"), ("lq_b", "LQ"))]
            with patch("wetland_coupling.maxent_engine.PROJECTION_CHUNK_ROWS", 17):
                results = run_jobs(jobs, max_workers=1)
            self.assertEqual([len(result["predictions"]) for result in results], [45, 45, 45])
            np.testing.assert_array_equal(results[0]["predictions"], results[2]["predictions"])
            for result in results:
                self.assertIsInstance(result["complexity"], int)
                self.assertGreater(result["complexity"], 0)
                manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
                self.assertFalse(manifest["background_is_absence"])
                self.assertFalse(manifest["output_is_calibrated_probability"])
                self.assertEqual(manifest["effective_maxent_rng_seed"], 0)
                self.assertEqual(manifest["status"], "OFFICIAL_MAXENT_FITTED")
                self.assertEqual(len(manifest["prediction_files"]), 3)
                self.assertIn("started_at_utc", manifest)


if __name__ == "__main__":
    unittest.main()
