"""Fail-closed input and fit-only sampling contracts with synthetic fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

from wetland_coupling.maxent_inputs import (
    EXPECTED_COUNTS, SEASONS, build_formal_inputs, dataframe_cells,
    sample_background, validate_raster, verify_hash,
)
from wetland_coupling.maxent_protocol import sha256


class MaxentInputContracts(unittest.TestCase):
    def setUp(self):
        self.grid = {"crs":"EPSG:32649","epsg":32649,"height":20,"width":20,"resolution_m":100,
                     "transform":[100.,0.,657500.,0.,-100.,3875600.]}
        self.env = np.arange(800,dtype="float32").reshape(20,20,2)/800
        self.names = ["a","b"]
        self.valid = np.ones((20,20),bool)
        self.fit = self.valid.copy()
        self.fit[:,10:] = False
        self.arrays = {"env":self.env,"names":self.names,"grid":self.grid,"valid_mask":self.valid}
        self.B0 = dataframe_cells(np.array([0,2,205,219]),self.env,self.names,self.grid)
        self.visits = dataframe_cells(np.array([0,45,219]),self.env,self.names,self.grid)
        self.visits["inferred_visit_count"] = [4,2,1000]
        self.config = {"background":{"is_absence":False,"uniform_points_per_fit":25,
                                       "kde_bandwidth_m":1000,"uniform_mixture_fraction":.1}}

    def test_deterministic_sampling_has_unique_cells_and_no_absence_semantics(self):
        for scheme in ("B0_target_group","B1_uniform","B2_visit_density_proxy"):
            a = sample_background(scheme,self.fit,self.arrays,self.visits,self.B0,self.config,42)
            b = sample_background(scheme,self.fit,self.arrays,self.visits,self.B0,self.config,42)
            pd.testing.assert_frame_equal(a,b)
            self.assertTrue(a.native_cell_id.is_unique)
            self.assertTrue((a.raster_col<10).all())
            self.assertTrue((a.background_is_absence==False).all())
            self.assertFalse(a.attrs["background_audit"]["background_is_absence"])

    def test_bias_never_reads_heldout_counts_and_cannot_use_locked_visits(self):
        first = sample_background("B2_visit_density_proxy",self.fit,self.arrays,self.visits,self.B0,self.config,73)
        altered = self.visits.copy()
        altered.loc[altered.raster_col>=10,"inferred_visit_count"] = np.nan
        second = sample_background("B2_visit_density_proxy",self.fit,self.arrays,altered,self.B0,self.config,73)
        pd.testing.assert_frame_equal(first,second)
        audit = first.attrs["background_audit"]
        self.assertEqual(audit["fit_inferred_visits"],6)
        self.assertEqual(audit["fit_visit_cells"],2)
        self.assertEqual(audit["excluded_nonfit_visit_cells"],1)
        self.assertFalse(audit["validation_or_locked_visits_used_for_bias"])
        self.assertFalse(audit["visit_effort_verified"])

    def test_no_fit_visits_is_explicit_uniform_fallback(self):
        visits = self.visits[self.visits.raster_col>=10]
        b2 = sample_background("B2_visit_density_proxy",self.fit,self.arrays,visits,self.B0,self.config,9)
        b1 = sample_background("B1_uniform",self.fit,self.arrays,visits,self.B0,self.config,9)
        np.testing.assert_array_equal(b2.native_cell_id,b1.native_cell_id)
        self.assertTrue(b2.attrs["background_audit"]["no_fit_visits_fallback_uniform"])

    def test_missing_predictor_nodata_wrong_mask_and_wrong_crs_fail_closed(self):
        with self.assertRaises(ValueError):
            dataframe_cells(np.array([0]),self.env,["a"],self.grid)
        env = self.env.copy()
        env[0,0,0] = np.nan
        with self.assertRaises(ValueError):
            dataframe_cells(np.array([0]),env,self.names,self.grid)
        with self.assertRaises(ValueError):
            dataframe_cells(np.array([0]),self.env,self.names,{**self.grid,"crs":"EPSG:3857"})
        with self.assertRaises(ValueError):
            sample_background("B1_uniform",self.fit[:5],self.arrays,self.visits,self.B0,self.config,1)
        badvalid = self.valid.copy()
        badvalid[0,0] = False
        with self.assertRaises(ValueError):
            sample_background("B1_uniform",self.fit,{**self.arrays,"valid_mask":badvalid},self.visits,self.B0,self.config,1)
        with self.assertRaises(ValueError):
            sample_background("B1_uniform",self.fit,self.arrays,self.visits,self.B0,{"background":{"is_absence":True}},1)

    def test_raster_hash_crs_nodata_and_alignment_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"fixture.tif"
            for crs,nodata in ((None,-9999),("EPSG:3857",-9999),("EPSG:32649",None)):
                self._raster(p,np.ones((3,4),"float32"),crs,nodata)
                with self.assertRaises(ValueError): validate_raster(p)
            self._raster(p,np.ones((3,4),"float32"),"EPSG:32649",-9999)
            grid,mask = validate_raster(p,expected_sha256=sha256(p))
            self.assertTrue(mask.all())
            with self.assertRaises(ValueError): verify_hash(p,"f"*64)
            with self.assertRaises(ValueError): validate_raster(p,{**grid,"width":5})

    @staticmethod
    def _raster(path,array,crs,nodata):
        with rasterio.open(path,"w",driver="GTiff",height=array.shape[0],width=array.shape[1],count=1,
              dtype="float32",crs=crs,nodata=nodata,transform=Affine(100,0,657500,0,-100,3875600)) as ds:
            ds.write(array.astype("float32"),1)

    def test_builder_preserves_common_domain_and_deduplicates_event_not_species(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"configs").mkdir()
            candidate = root/"local_work/candidates"
            candidate.mkdir(parents=True)
            obs = root/"local_work/prepared_observations_002"
            obs.mkdir()
            envroot = root/"local_work/env"
            scope = root/"configs/scope.json"
            scope.write_text(json.dumps({"status":"FAMILY_SCOPE_CANDIDATE_PENDING_SPECIES_REVIEW"}))
            grid = {**self.grid,"height":3,"width":4}
            center = dataframe_cells(np.array([0,1]),np.ones((3,4,1),"float32"),["x"],grid)
            records = []
            for season in SEASONS:
                for event,cell in (("visit_a",0),("visit_a",0),("visit_b",0),("visit_c",1)):
                    records.append({"survey_event_key_unconfirmed":season+event,"abundance_numeric":1,
                        "date_start_parsed":"2024-03-01","season_calendar":season,
                        "longitude_numeric_unconfirmed_crs":center.longitude.iloc[cell],
                        "latitude_numeric_unconfirmed_crs":center.latitude.iloc[cell]})
            ledger = root/"local_work/ledger.csv"
            pd.DataFrame(records).to_csv(ledger,index=False)
            observation = {"coordinate_reference_confirmed":True,"confirmed_source_crs":"EPSG:4326",
                "input_ledger_sha256":sha256(ledger),"scope_sha256":sha256(scope)}
            (obs/"manifest.json").write_text(json.dumps(observation))
            source = {"source_observation_manifest_sha256":sha256(obs/"manifest.json"),"seasons":{},"outputs":{}}
            for season,subdir in SEASONS.items():
                dest = envroot/subdir
                dest.mkdir(parents=True)
                names = [f"cov_{n}" for n in range(EXPECTED_COUNTS[season])]
                arrays = np.stack([np.arange(12).reshape(3,4)+n for n in range(len(names))],axis=2).astype("float32")
                if season=="summer": arrays[2,3,0] = -9999
                paths = []
                for n,name in enumerate(names):
                    p = dest/(name+".tif")
                    self._raster(p,arrays[:,:,n],"EPSG:32649",-9999)
                    paths.append({"path":str(p),"sha256":sha256(p)})
                source["seasons"][season] = {"environment_sources":paths}
                for prefix in ("occurrence_cells_","target_group_background_cells_"):
                    p = candidate/(prefix+season+".csv")
                    frame = dataframe_cells(np.array([0,1]),arrays,names,grid)
                    frame.to_csv(p,index=False)
                    source["outputs"][p.name] = {"sha256":sha256(p)}
            (candidate/"manifest.json").write_text(json.dumps(source))
            config = {"protocol_version":"synthetic_input_contract","seed":123,"scope_policy":"configs/scope.json",
                "background":{"is_absence":False,"uniform_points_per_fit":4},
                "inputs":{"candidate_tables":"local_work/candidates","environment":"local_work/env","bird_ledger":"local_work/ledger.csv"}}
            cp = root/"configs/formal.json"
            cp.write_text(json.dumps(config))
            with patch("wetland_coupling.maxent_inputs.subprocess.check_output",return_value="fixture_git_sha"):
                first = build_formal_inputs(cp,root/"inputs_a",root)
                second = build_formal_inputs(cp,root/"inputs_b",root)
            self.assertEqual(first["common_valid_cells"],11)
            self.assertFalse(first["strict_end_to_end_oof"])
            self.assertFalse(first["gate_eligible"])
            self.assertFalse(first["B2_generated_globally"])
            visits = pd.read_csv(root/"inputs_a/spring/visit_cells.csv")
            self.assertEqual(visits.inferred_visit_count.tolist(),[2,1])
            np.testing.assert_array_equal(np.load(root/"inputs_a/common_valid_mask.npy"),np.load(root/"inputs_b/common_valid_mask.npy"))
            for season in SEASONS:
                self.assertEqual(sha256(root/f"inputs_a/{season}/B1.csv"),sha256(root/f"inputs_b/{season}/B1.csv"))
                self.assertEqual(sha256(root/f"inputs_a/{season}/env.npy"),sha256(root/f"inputs_b/{season}/env.npy"))
            for name,detail in first["outputs"].items():
                self.assertEqual(sha256(root/"inputs_a"/name),detail["sha256"])
            with self.assertRaises(FileExistsError): build_formal_inputs(cp,root/"inputs_a",root)


if __name__=="__main__":
    unittest.main()
