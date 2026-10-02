import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wetland_coupling.audit import audit_provenance, build_features, load_table, validate_table
from wetland_coupling.demo import make_demo
from wetland_coupling.fusion import fuse, geometric, propagate_paired, quadrants, relative_degradation
from wetland_coupling.model import DualGate
from wetland_coupling.parameters import sample_prior, rank_calibration_candidates, make_run_plan
from wetland_coupling.rasters import fuse_rasters
from wetland_coupling.training import load_gate, predict_with_model, train, propagate_gate_paired
from wetland_coupling.splits import base_crossfit_plan


class FusionContracts(unittest.TestCase):
    def test_intrinsically_low_suitability_is_not_degradation(self):
        self.assertAlmostEqual(float(relative_degradation(.3,.3)),0)
        self.assertTrue(np.isnan(relative_degradation(0,0)))
        r = fuse([.9],[.3],feasible=[1],habitat_suitability=[.3],deficit_mode="relative_degradation")
        self.assertEqual(r["restoration_candidate"][0],0)

    def test_geometry_vs_linear_and_repair(self):
        geo = fuse([.9],[.1],feasible=[1],habitat_suitability=[1])
        linear = fuse([.9],[.1],method="linear",feasible=[1],habitat_suitability=[1])
        self.assertAlmostEqual(geo["conservation"][0],.3)
        self.assertAlmostEqual(linear["conservation"][0],.5)
        self.assertAlmostEqual(geo["restoration_candidate"][0],.9)

    def test_zero_and_endpoint_weights(self):
        self.assertEqual(float(geometric(0,.5,.5)),0)
        self.assertEqual(float(geometric(0,.5,0)),.5)
        self.assertEqual(float(geometric(.7,0,1)),.7)

    def test_nodata_is_not_zero_quality(self):
        self.assertTrue(np.isnan(geometric(np.nan,.5)))
        out = fuse([.9],[np.nan],feasible=[1],habitat_suitability=[1])
        self.assertTrue(np.isnan(out["restoration_candidate"][0]))

    def test_invalid_values_rejected(self):
        for a in (-.01,1.01,np.inf):
            with self.assertRaises(ValueError):
                geometric(a,.5)

    def test_restoration_requires_explicit_masks(self):
        with self.assertRaises(ValueError):
            fuse(.9,.1)
        out = fuse([.9,.9],[0,.1],feasible=[1,0],habitat_suitability=[0,1])
        np.testing.assert_array_equal(out["restoration_candidate"],[0,0])

    def test_management_quadrants_and_review(self):
        out = quadrants([.9,.9,.2,.2,.9],[.9,.1,.9,.1,0],[1,1,1,1,1],[1,1,1,1,0])
        np.testing.assert_array_equal(out,[1,2,3,4,5])

    def test_geometry_shortfall_bounds(self):
        rng = np.random.default_rng(3)
        a,b,w = rng.random((3,100))
        g = geometric(a,b,w)
        self.assertTrue(np.all(g<=w*a+(1-w)*b+1e-12))
        self.assertTrue(np.all(g>=np.minimum(a,b)-1e-12))

    def test_uncertainty_is_propagated_not_mean_fused(self):
        m = np.array([[.1],[.9]])
        h = np.array([[.9],[.1]])
        out = propagate_paired(m,h,[1],[1])
        self.assertAlmostEqual(out["conservation_median"][0],.3)
        self.assertNotAlmostEqual(out["conservation_median"][0],float(geometric(m.mean(),h.mean())))

    def test_uncertainty_incomplete_members_fail_closed(self):
        out = propagate_paired([[.1],[np.nan]],[[.9],[.1]],[1],[1])
        self.assertTrue(np.isnan(out["conservation_median"][0]))
        self.assertTrue(np.isnan(out["zone_agreement"][0]))


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        make_demo(cls.root/"demo")
        cls.df = load_table(cls.root/"demo/samples.csv")
        cls.manifest = json.loads((cls.root/"demo/provenance.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_declared_provenance_pass(self):
        self.assertEqual(audit_provenance(self.df,self.manifest,True)["status"],"DECLARED_PROVENANCE_PASS")

    def test_synthetic_not_used_as_real(self):
        with self.assertRaises(ValueError):
            audit_provenance(self.df,self.manifest)

    def test_gate_validation_leak_rejected(self):
        bad = copy.deepcopy(self.manifest)
        val = self.df[self.df.role=="validation"].iloc[0]
        bad["artifacts"]["M_00"]["tune_sample_ids"] = [val.sample_id]
        bad["artifacts"]["M_00"]["tune_group_ids"] = [val.group_id]
        with self.assertRaisesRegex(ValueError,"leaked"):
            audit_provenance(self.df,bad,True)

    def test_mixed_maxent_scale_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad["artifacts"]["M_00"]["output_type"] = "logistic"
        with self.assertRaisesRegex(ValueError,"Mixed MaxEnt"):
            audit_provenance(self.df,bad,True)

    def test_nested_plan_excludes_gate_validation_and_test(self):
        plan = base_crossfit_plan(self.df)
        protected = set(self.df.loc[self.df.role!="train","sample_id"])
        for item in plan["fits"]:
            source = set(item["base_source_sample_ids"])
            self.assertFalse(source & protected)
            self.assertFalse(source & set(item["prediction_sample_ids"]))
            for fold in item["inner_folds"]:
                self.assertFalse(set(fold["fit_sample_ids"]) & set(fold["validation_sample_ids"]))

    def test_missing_restoration_supervision_keeps_head_untrained(self):
        temp = self.root/"no_repair_supervision"
        temp.mkdir()
        df = self.df.copy()
        df["target_restoration"] = np.nan
        df.to_csv(temp/"samples.csv",index=False)
        from wetland_coupling.audit import file_sha256
        prov = copy.deepcopy(self.manifest)
        prov["table_sha256"] = file_sha256(temp/"samples.csv")
        prov["supervision"].pop("restoration")
        (temp/"provenance.json").write_text(json.dumps(prov))
        report = train(temp/"samples.csv",temp/"provenance.json",temp/"trained",{"epochs":15},True)
        self.assertEqual(report["active_heads"],[True,False])
        predictions = pd.read_csv(temp/"trained/predictions.csv")
        np.testing.assert_array_equal(predictions.gamma,.5*np.ones(len(predictions)))

    def test_fold_group_leak_rejected_even_if_id_missing(self):
        bad = copy.deepcopy(self.manifest)
        bad["artifacts"]["M_00"]["fit_group_ids"].append("block_00")
        with self.assertRaisesRegex(ValueError,"group-out-of-fold"):
            audit_provenance(self.df,bad,True)

    def test_fusion_pseudolabel_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad["supervision"]["restoration"]["derived_from_m_or_h"] = True
        with self.assertRaisesRegex(ValueError,"pseudo-labels"):
            audit_provenance(self.df,bad,True)

    def test_duplicate_sampling_unit_rejected(self):
        bad = pd.concat([self.df,self.df.iloc[:1]],ignore_index=True)
        bad.loc[len(bad)-1,"sample_id"] = "new_id_same_site"
        with self.assertRaisesRegex(ValueError,"Duplicate site"):
            validate_table(bad)

    def test_bounded_gate_head_and_gradient(self):
        torch.manual_seed(1)
        model = DualGate(dropout=0)
        x = torch.tensor(build_features(self.df.iloc[:20]))
        m = torch.tensor(self.df.m.iloc[:20].to_numpy(),dtype=torch.float32)
        h = torch.tensor(self.df.h.iloc[:20].to_numpy(),dtype=torch.float32)
        pred,w = model(x,m,h,torch.ones(20,dtype=torch.bool))
        np.testing.assert_allclose(w.detach().sum(-1),1,atol=1e-6)
        self.assertTrue(bool((w>=.1).all() & (w<=.9).all()))
        pred.sum().backward()
        self.assertGreater(float(model.head.weight.grad.abs().sum()),0)

    def test_inactive_head_is_fixed_half(self):
        model = DualGate(active_heads=(True,False))
        with torch.no_grad():
            model.head.bias[:] = torch.tensor([1.,-1.,2.,-2.])
        w = model.weights(torch.tensor(build_features(self.df.iloc[:3])))
        np.testing.assert_array_equal(w.detach().numpy()[:,1,:],.5*np.ones((3,2)))

    def test_gate_gradient_matches_finite_difference(self):
        torch.manual_seed(2)
        model = DualGate(dropout=0).double().eval()
        x = torch.rand(8,6,dtype=torch.float64)
        m,h = torch.full((8,),.8,dtype=torch.float64),torch.full((8,),.2,dtype=torch.float64)
        e = torch.ones(8,dtype=torch.bool)
        pred,_ = model(x,m,h,e)
        pred[:,0].sum().backward()
        analytical = float(model.head.bias.grad[0])
        vals = []
        with torch.no_grad():
            for delta in (-1e-5,1e-5):
                model.head.bias[0] = delta
                vals.append(float(model(x,m,h,e)[0][:,0].sum()))
            model.head.bias[0] = 0
        self.assertAlmostEqual(analytical,(vals[1]-vals[0])/2e-5,places=6)

    def test_train_save_reload_and_uncertainty(self):
        out = self.root/"training"
        report = train(self.root/"demo/samples.csv",self.root/"demo/provenance.json",out,
                       {"epochs":90,"dropout":0.0},True)
        self.assertEqual(report["status"],"SYNTHETIC_DEMO_ONLY")
        model,mean,scale,meta = load_gate(out/"gate.pt")
        pred,_ = predict_with_model(model,self.df,mean,scale)
        saved = pd.read_csv(out/"predictions.csv")
        np.testing.assert_allclose(pred[:,0],saved.conservation_score,atol=1e-6)
        self.assertTrue(report["metrics"]["test"]["dual_gate"]["target_protection"]["n"]>0)
        subset = self.df.iloc[:5]
        m = np.vstack([subset.m]*3)
        h = np.vstack([subset.h]*3)
        propagated = propagate_gate_paired(out/"gate.pt",subset,m,h,demo=True)
        np.testing.assert_allclose(propagated["conservation_median"],pred[:5,0],atol=1e-6)

    def test_raster_real_file_alignment_and_nodata(self):
        import rasterio
        inputs = {k:self.root/f"demo/raster_inputs/{k}.tif" for k in ("m","h","feasible","habitat_suitability")}
        report = fuse_rasters(inputs,self.root/"raster_output")
        self.assertEqual(report["valid_pixels"],31*40)
        with rasterio.open(self.root/"raster_output/conservation_score.tif") as ds:
            a = ds.read(1,masked=True)
            self.assertTrue(a.mask[0].all())
            self.assertEqual(ds.res,(100,100))
        with rasterio.open(self.root/"raster_output/restoration_candidate_score.tif") as ds:
            a = ds.read(1)
            self.assertTrue((a[1:4,20:30]==0).all())
            self.assertTrue((a[27:,30:]==0).all())

    def test_raster_shift_rejected(self):
        import rasterio
        from affine import Affine
        srcpath = self.root/"demo/raster_inputs/m.tif"
        shifted = self.root/"shifted.tif"
        with rasterio.open(srcpath) as ds:
            profile = ds.profile.copy()
            profile["transform"] = Affine.translation(50,0)*profile["transform"]
            with rasterio.open(shifted,"w",**profile) as dst:
                dst.write(ds.read())
        inputs = {k:self.root/f"demo/raster_inputs/{k}.tif" for k in ("m","h","feasible","habitat_suitability")}
        inputs["m"] = shifted
        with self.assertRaisesRegex(ValueError,"origin"):
            fuse_rasters(inputs,self.root/"shift_should_fail")


class ParameterContracts(unittest.TestCase):
    def test_run_plan_keeps_metres_and_resolves_original_threat_paths(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            spec = {"status":"evidence_reviewed","target_invest_version":"3.16.1","distance_unit":"m",
                    "fixed_half_saturation_constant":.5,
                    "parameters":[{"key":"weight:urban","low":.5,"high":1.,"evidence":"synthetic://test_only"}]}
            (p/"prior.json").write_text(json.dumps(spec))
            pd.DataFrame({"threat":["urban"],"max_dist":[1500.],"weight":[1.],
                          "decay":["linear"],"cur_path":["threat.tif"]}).to_csv(p/"threats.csv",index=False)
            pd.DataFrame({"lucode":[1],"habitat":[1.],"urban":[.5]}).to_csv(p/"sensitivity.csv",index=False)
            import rasterio
            from rasterio.transform import from_origin
            for name, dtype, value in (("lulc.tif", "uint8", 1), ("threat.tif", "float32", .5)):
                with rasterio.open(p/name, "w", driver="GTiff", width=2, height=2,
                                   count=1, dtype=dtype, crs="EPSG:32649",
                                   transform=from_origin(657500, 3875600, 100, 100)) as dst:
                    dst.write(np.full((2, 2), value, dtype=dtype), 1)
            plan = make_run_plan(p/"prior.json",p/"threats.csv",p/"sensitivity.csv",p/"lulc.tif",p/"plan",2)
            self.assertEqual(plan["status"],"PRIOR_RUN_PLAN_ONLY")
            tt = pd.read_csv(p/"plan/invest_0000/threats.csv")
            self.assertEqual(tt.max_dist.iloc[0],1500.)
            self.assertEqual(tt.cur_path.iloc[0],str((p/"threat.tif").resolve()))

    def test_candidate_calibration_uses_identical_inner_folds(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            rows = [{"member_id":m,"fold":f,"score":(.8 if m=="a" else .5)+.01*f,
                     "metric":"Spearman","higher_is_better":1,"split_role":"inner_calibration_validation"}
                    for m in ["a","b"] for f in range(3)]
            pd.DataFrame(rows).to_csv(p/"input.csv",index=False)
            result = rank_calibration_candidates(p/"input.csv",p/"out.csv")
            self.assertEqual(result["best_member"],"a")

    def test_missing_evidence_blocks_prior(self):
        with self.assertRaises(ValueError):
            sample_prior({"status":"pending_evidence"})

    def test_prior_samples_repeatable_and_metres_only(self):
        spec = {"status":"evidence_reviewed","target_invest_version":"3.16.1","distance_unit":"m",
                "parameters":[{"key":"weight:urban","low":.5,"high":1.,"evidence":"synthetic://test_only"}]}
        pd.testing.assert_frame_equal(sample_prior(spec),sample_prior(spec))
        spec["distance_unit"] = "km"
        with self.assertRaisesRegex(ValueError,"metres"):
            sample_prior(spec)

    def test_final_test_cannot_select_invest_parameters(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/"scores.csv"
            pd.DataFrame({"member_id":["a"],"fold":[0],"score":[.9],"metric":["Spearman"],
                          "higher_is_better":[1],"split_role":["test"]}).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,"outer/test"):
                rank_calibration_candidates(p,Path(t)/"out.csv")


if __name__ == "__main__":
    unittest.main()
