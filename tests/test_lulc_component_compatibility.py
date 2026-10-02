import unittest
import numpy as np
from scripts.zhengzhou.audit_lulc_component_compatibility import compatibility, FAMILIES


class LulcCompatibilityContracts(unittest.TestCase):
    def test_perfect_numerical_match_is_still_not_semantic_confirmation(self):
        summary,table,match=compatibility(np.arange(1,10),np.eye(9))
        self.assertTrue(summary['component_values_probability_like'])
        self.assertEqual(summary['max_assignment_agreement_fraction'],1.)
        np.testing.assert_array_equal(table,np.eye(9,dtype=int))
        self.assertEqual([r['hypothesized_component_descriptor'] for r in match],list(FAMILIES))
        self.assertTrue(all(r['not_ground_truth_accuracy'] and not r['class_semantics_confirmed'] for r in match))

    def test_incomplete_component_vectors_do_not_produce_a_crosswalk(self):
        summary,_,match=compatibility(np.arange(1,10),np.eye(9)*.8)
        self.assertFalse(summary['component_values_probability_like'])
        self.assertIsNone(summary['max_assignment_agreement_fraction'])
        self.assertEqual(match,[])

    def test_fractional_category_codes_are_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError,'Invalid compatibility inputs'):
            compatibility(np.array([1.,2.5]),np.eye(9)[:2])

    def test_missing_component_values_fail_closed(self):
        values=np.eye(9);values[0,1]=np.nan
        with self.assertRaisesRegex(ValueError,'Invalid compatibility inputs'):compatibility(np.arange(1,10),values)


if __name__=='__main__':unittest.main()
