import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from wetland_coupling.maxent_protocol import (background_auc, validate_predictions, select_predictors,
    metrics, choose_candidate, assert_oof_groups, freeze_selection, claim_locked_test)


class MaxentProtocolTests(unittest.TestCase):
    def test_background_auc_is_rank_statistic_not_absence_accuracy(self):
        self.assertEqual(background_auc([0.8,0.9],[0.1,0.2]),1)
        self.assertEqual(background_auc([0.5],[0.5]),0.5)
        self.assertEqual(background_auc([0.1],[0.9]),0)
        self.assertFalse(metrics([0.8,0.9],[0.1,0.2],[0.5],[0.5],2)["background_is_absence"])

    def test_scale_fails_closed(self):
        for values in ([np.nan],[-0.1],[1.01],[np.inf]):
            with self.assertRaises(ValueError): validate_predictions(values)

    def test_predictor_selection_is_deterministic_fit_only(self):
        frame=pd.DataFrame({"a":[1,2,3,4],"b":[2,4,6,8],"c":[1,1,1,1]})
        self.assertEqual(select_predictors(frame,["c","b","a"])[0],["a"])
        self.assertEqual(select_predictors(frame,["a","b","c"]),select_predictors(frame,["c","b","a"]))
        with self.assertRaises(ValueError): select_predictors(frame,["missing"])

    def test_own_and_locked_groups_excluded(self):
        assert_oof_groups(["v"],["a"],["a"],["t"])
        with self.assertRaises(ValueError): assert_oof_groups(["v"],["v"],[],["t"])
        with self.assertRaises(ValueError): assert_oof_groups(["v"],["a"],["t"],["t"])

    def test_frozen_selection_and_test_once(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"frozen.json"
            h=freeze_selection(path,{"rm":1},"split","config")
            claim_locked_test(d,path,h,"split","config")
            with self.assertRaises(FileExistsError): claim_locked_test(d,path,h,"split","config")
            with self.assertRaises(FileExistsError): freeze_selection(path,{},"split","config")
            with self.assertRaises(ValueError): claim_locked_test(d,path,h,"wrong","config")

    def test_one_se_selection_uses_complexity_after_omission(self):
        rows=[dict(variant="no_lst",background="B1",rm=1,fc="LQ",mean_auc=.85,auc_se=.04,mean_omission=.1,mean_complexity=12,mean_abs_auc_gap=.1),
              dict(variant="no_lst",background="B1",rm=4,fc="L",mean_auc=.82,auc_se=.02,mean_omission=.1,mean_complexity=3,mean_abs_auc_gap=.02),
              dict(variant="static_lst",background="B1",rm=.5,fc="LQH",mean_auc=.95,auc_se=.01,mean_omission=.5,mean_complexity=30,mean_abs_auc_gap=.2)]
        self.assertEqual(choose_candidate(rows)["rm"],4)


if __name__ == "__main__": unittest.main()
