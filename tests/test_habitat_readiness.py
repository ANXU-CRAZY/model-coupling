import unittest

import numpy as np

from scripts.zhengzhou.audit_habitat_readiness import (
    check_values, conservative_support_distance, coverage_row, crosswalk_missing,
)


class HabitatReadinessContracts(unittest.TestCase):
    def test_zero_threat_is_valid_and_missing_is_not_zero(self):
        values = np.array([[0., .5, -9999.]])
        valid = np.array([[True, True, False]])
        result = check_values(values, valid, "threat")
        self.assertEqual(result["valid_pixels"], 2)
        self.assertEqual(result["zero_pixels"], 1)
        coverage = coverage_row("example", valid, np.ones_like(valid), valid)
        self.assertEqual(coverage["missing_in_common"], 1)

    def test_lulc_fractional_code_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "integer category"):
            check_values(np.array([[1., 2.1]]), np.ones((1, 2), dtype=bool), "lulc")

    def test_threat_out_of_range_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            check_values(np.array([[1.01]]), np.ones((1, 1), dtype=bool), "threat")

    def test_padding_treats_full_raster_outside_as_unknown(self):
        mask = np.ones((11, 11), dtype=bool)
        distances = conservative_support_distance(mask, 100.)
        self.assertEqual(distances[0, 5], 0.)
        self.assertAlmostEqual(distances[5, 5], 600. - np.sqrt(2.) * 100.)
        self.assertFalse((distances >= 500.).any())

    def test_internal_nodata_hole_limits_buffer_support(self):
        mask = np.ones((21, 21), dtype=bool)
        before = conservative_support_distance(mask, 100.)
        mask[10, 10] = False
        after = conservative_support_distance(mask, 100.)
        self.assertEqual(after[10, 10], 0.)
        self.assertLess(after[10, 11], before[10, 11])
        np.testing.assert_array_less(after, before + 1.e-10)

    def test_class_name_without_source_evidence_is_insufficient(self):
        spec = {"class_crosswalk": [{"lucode": 1, "name": "unverified guess",
                                      "status": "SOURCE_CROSSWALK_CONFIRMED"}]}
        self.assertEqual(crosswalk_missing([1, 2], spec), [1, 2])
        spec["class_crosswalk"][0]["evidence"] = "verified producer legend"
        self.assertEqual(crosswalk_missing([1, 2], spec), [2])

    def test_duplicate_crosswalk_code_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            crosswalk_missing([1], {"class_crosswalk": [{"lucode": 1}, {"lucode": 1}]})


if __name__ == "__main__":
    unittest.main()
