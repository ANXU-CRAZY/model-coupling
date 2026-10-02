import unittest

import numpy as np
from scipy.optimize import check_grad

from wetland_coupling.parameter_diagnostics import (
    fit_logistic, logistic_objective, ols_vif, rank_correlation,
)
from scripts.zhengzhou.audit_parameter_derivation_phase1 import phase_c_threat_overlap
from scripts.zhengzhou.derive_habitat_parameters_phase2 import logistic_irls, vif
from scripts.zhengzhou.audit_parameter_identifiability_phase2c import logistic_w, rank_corr
from scripts.zhengzhou.derive_hj_profile_phase3 import logistic


class ExploratoryDiagnosticContracts(unittest.TestCase):
    def test_phase1_spearman_handles_ties(self):
        self.assertAlmostEqual(rank_correlation([0, 0, 1, 2], [0, 1, 0, 2]), .5)

    def test_phase1_actual_overlap_function_uses_average_tied_ranks(self):
        from unittest.mock import patch
        from scripts.zhengzhou import audit_parameter_derivation_phase1 as module
        def layer(path):
            values = [0, 1, 0, 2] if path.name == "nightlight_spring.tif" else [0, 0, 1, 2]
            return np.array(values, dtype=float).reshape(2, 2), None, (2, 2)
        with patch.object(module, "read_raster", side_effect=layer), patch("pathlib.Path.exists", return_value=True):
            result = module.phase_c_threat_overlap()
        self.assertEqual(result["threat_vs_environment_spearman"]["night_light|env_nightlight"], .5)

    def test_constant_ranking_is_unknown(self):
        self.assertIsNone(rank_correlation([1, 1, 1], [1, 2, 3]))

    def test_vif_uses_joint_ols_not_sum_pairwise_correlations(self):
        rng = np.random.default_rng(4)
        z = rng.normal(size=(200, 3))
        x = np.column_stack([z[:, 0] + z[:, 1], z[:, 0] + .4 * z[:, 2], z[:, 0] + .2 * z[:, 1]])
        result = ols_vif(x)
        corr = np.corrcoef(x, rowvar=False)
        incorrect = 1. / max(1. - (corr[0, 1] ** 2 + corr[0, 2] ** 2), 1.e-9)
        self.assertNotAlmostEqual(result[0], incorrect, places=2)
        standardized = (x - x.mean(axis=0)) / x.std(axis=0)
        self.assertAlmostEqual(result[0], np.linalg.inv(np.corrcoef(standardized, rowvar=False))[0, 0])

    def test_perfect_dependence_does_not_fabricate_finite_vif(self):
        self.assertEqual(ols_vif(np.array([[1, 2], [2, 4], [3, 6.]])), [None, None])

    def test_phase2_logistic_score_matches_numerical_gradient(self):
        x = np.column_stack([np.ones(6), [-2, -1, 0, 0, 1, 2]])
        y, weights, beta = np.array([0, 1, 0, 1, 0, 1.]), np.array([1, 3, 2, 4, 1, 2.]), np.array([.2, -.3])
        error = check_grad(lambda b: logistic_objective(b, x, y, weights, .01)[0],
                           lambda b: logistic_objective(b, x, y, weights, .01)[1], beta)
        self.assertLess(error, 1.e-6)

    def test_all_four_script_numeric_interfaces_use_checked_helpers(self):
        x = np.ones((10, 1))
        y = np.array([1.] * 2 + [0.] * 8)
        self.assertAlmostEqual(logistic_irls(x, y)[0], np.log(.25), places=5)
        self.assertAlmostEqual(logistic_w(x, y, np.array([4.] * 2 + [1.] * 8))[0], 0., places=5)
        self.assertAlmostEqual(logistic(x, y)[0], np.log(.25), places=5)
        self.assertAlmostEqual(rank_corr(np.array([0, 0, 1, 2]), np.array([0, 1, 0, 2])), .5)
        self.assertTrue(callable(phase_c_threat_overlap))

    def test_phase3_report_writer_remains_in_main_not_an_unreachable_helper(self):
        import ast
        import inspect
        from scripts.zhengzhou import derive_hj_profile_phase3 as module
        tree = ast.parse(inspect.getsource(module))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertTrue(any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                            and isinstance(n.value.func, ast.Name) and n.value.func.id == "save_report"
                            for n in main.body))

    def test_phase2c_weights_really_change_the_fitted_intercept(self):
        x = np.ones((10, 1))
        y = np.array([1.] * 2 + [0.] * 8)
        unweighted = fit_logistic(x, y)
        weighted = fit_logistic(x, y, np.array([4.] * 2 + [1.] * 8))
        self.assertAlmostEqual(unweighted[0], np.log(.25), places=5)
        self.assertAlmostEqual(weighted[0], 0., places=5)

    def test_balancing_each_class_membership_erases_class_contrast(self):
        codes = np.array([0] * 10 + [1] * 10)
        y = np.array([1.] * 2 + [0.] * 8 + [1.] * 8 + [0.] * 2)
        weights = np.zeros(len(y))
        for code in (0, 1):
            for membership in (0, 1):
                mask = (codes == code) & (y == membership)
                weights[mask] = .5 / mask.sum()
        beta = fit_logistic(np.column_stack([np.ones(len(y)), codes]), y, weights)
        np.testing.assert_allclose(beta, 0., atol=1.e-6)

    def test_phase3_classifier_normalization_depends_on_intercept(self):
        from scipy.special import expit
        a = expit(np.array([-2., 1.])); b = expit(np.array([-2., 1.]) + 3.)
        self.assertNotAlmostEqual((a / a.max())[0], (b / b.max())[0])

    def test_nonfinite_predictor_and_invalid_weights_fail_closed(self):
        with self.assertRaises(ValueError):
            fit_logistic([[1, np.nan], [1, 2]], [0, 1])
        with self.assertRaises(ValueError):
            fit_logistic([[1], [1]], [0, 1], [0, 1])

    def test_iteration_failure_is_not_success(self):
        with self.assertRaisesRegex(ValueError, "convergence"):
            fit_logistic(np.column_stack([np.ones(6), np.arange(6)]), [0, 1, 0, 1, 1, 1], max_iter=0)

    def test_four_phases_refuse_existing_output_before_reading_data(self):
        import tempfile
        from unittest.mock import patch
        from scripts.zhengzhou import (
            audit_parameter_derivation_phase1 as phase1,
            derive_habitat_parameters_phase2 as phase2,
            audit_parameter_identifiability_phase2c as phase2c,
            derive_hj_profile_phase3 as phase3,
        )
        with tempfile.TemporaryDirectory() as temp:
            for module in (phase1, phase2, phase2c, phase3):
                argv = ["phase", "--out", temp]
                if module is phase3:
                    argv += ["--comparison-audit", "unopened.json"]
                with self.subTest(module=module.__name__), patch("sys.argv", argv), self.assertRaises(FileExistsError):
                    module.main()


if __name__ == "__main__":
    unittest.main()
