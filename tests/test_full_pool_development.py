import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.zhengzhou import run_full_pool_development as runner
from scripts.zhengzhou.audit_full_pool_development import prediction_values, validate_cleanup_receipt, validate_source_snapshot
from wetland_coupling.maxent_protocol import sha256, write_json
from wetland_coupling.maxent_splits import build_spatial_split_plan, save_spatial_split_plan


class FullPoolDevelopmentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid={'crs':'EPSG:32649','height':60,'width':60,'transform':[100.,0.,650000.,0.,-100.,3870000.]}
        r,c=np.nonzero(np.indices((60,60))[0]%2==0)
        table=pd.DataFrame({'native_cell_id':[f'r{a}c{b}' for a,b in zip(r,c)],'raster_row':r,'raster_col':c,
            'x_utm49':650000.+(c+.5)*100,'y_utm49':3870000.-(r+.5)*100})
        cls.config={'seed':32,'spatial':{'block_size_m':1000,'buffer_m':100,'outer_folds':3,'inner_folds':3,
            'locked_test_fraction_target':.2,'assignment_seed_trials':8,'minimum_fit_presences':20,
            'minimum_validation_presences':10,'minimum_background':10}}
        presences={s:table.copy() for s in runner.SEASONS}
        cls.plan=build_spatial_split_plan(presences,{s:{'B0':table.copy(),'B1':table.copy()} for s in runner.SEASONS},
            cls.grid,np.ones((60,60),bool),cls.config)

    def test_partition_loader_never_deserializes_bird_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'split';save_spatial_split_plan(self.plan,root,self.config)
            with patch.object(pd,'read_csv',side_effect=AssertionError('Bird responses must not be read')):
                actual=runner.development_split_plan(root)
            self.assertEqual(actual['split_hash'],self.plan['split_hash'])
            np.testing.assert_array_equal(actual['development_fit_mask'],self.plan['development_fit_mask'])
            for fold in actual['outer_folds']:
                self.assertFalse(np.any(fold['fit_mask']&actual['locked_mask']))

    def test_partition_loader_fails_on_changed_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'split';save_spatial_split_plan(self.plan,root,self.config)
            with (root/'split_masks.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):runner.development_split_plan(root)

    def fixture(self,directory):
        run=Path(directory)/'run';scope=run/'scope';model=scope/'official_model';chunks=model/'projection_inputs'
        chunks.mkdir(parents=True)
        for name in ('train.csv','background.csv','projection.csv'):(scope/name).write_text(name,encoding='utf-8')
        (chunks/'projection_part00000.csv').write_text('reconstructable',encoding='utf-8')
        (model/'model.lambdas').write_text('retained lambda',encoding='utf-8')
        manifest=model/'manifest.json'
        data={'status':'OFFICIAL_MAXENT_FITTED','inputs':{key:{'sha256':sha256(scope/(key+'.csv'))} for key in ('train','background','projection')},
            'input_paths':{key+'_csv':str(scope/(key+'.csv')) for key in ('train','background','projection')},
            'outputs':{str(p.relative_to(model)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in model.rglob('*') if p.is_file()}}
        write_json(manifest,data)
        return run,scope,manifest

    def test_cleanup_removes_only_reconstructed_projections_and_accounts_for_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            run,scope,manifest=self.fixture(directory)
            runner.cleanup_projection(scope,[manifest],run)
            self.assertFalse((scope/'projection.csv').exists())
            self.assertFalse((manifest.parent/'projection_inputs/projection_part00000.csv').exists())
            for name in ('train.csv','background.csv'):self.assertTrue((scope/name).is_file())
            self.assertTrue((manifest.parent/'model.lambdas').is_file())
            self.assertEqual(runner.load(scope/'projection_cleanup.json')['status'],'REMOVED')
            runner.audit_engine(manifest)
            runner.cleanup_projection(scope,[manifest],run)

    def test_missing_output_without_receipt_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            _,_,manifest=self.fixture(directory);(manifest.parent/'model.lambdas').unlink()
            with self.assertRaisesRegex(ValueError,'Unaccounted missing output'):runner.audit_engine(manifest)

    def test_cleanup_receipt_is_independently_bound_to_projection_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            run,scope,manifest=self.fixture(directory);runner.cleanup_projection(scope,[manifest],run)
            receipt=validate_cleanup_receipt(scope,[manifest])
            self.assertGreater(receipt['bytes'],0)

    def test_forged_receipt_cannot_account_for_model_lambda_as_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            run,scope,manifest=self.fixture(directory);runner.cleanup_projection(scope,[manifest],run)
            receipt=runner.load(scope/'projection_cleanup.json');path=manifest.parent/'model.lambdas'
            receipt['files'].append({'path':str(path.resolve()),'sha256':sha256(path),'bytes':path.stat().st_size})
            receipt['bytes']=sum(r['bytes'] for r in receipt['files']);write_json(scope/'projection_cleanup.json',receipt)
            with self.assertRaisesRegex(ValueError,'not a verified projection input'):validate_cleanup_receipt(scope,[manifest])

    def test_changed_cleanup_projection_hash_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            run,scope,manifest=self.fixture(directory);runner.cleanup_projection(scope,[manifest],run)
            receipt=runner.load(scope/'projection_cleanup.json');receipt['files'][0]['sha256']='changed'
            write_json(scope/'projection_cleanup.json',receipt)
            with self.assertRaisesRegex(ValueError,'not a verified projection input'):validate_cleanup_receipt(scope,[manifest])

    def test_audit_source_change_between_start_and_finish_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'audit.py';source.write_text('frozen audit code',encoding='utf-8')
            recorded={'audit.py':sha256(source)};validate_source_snapshot(root,recorded)
            source.write_text('different audit code',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Audit source changed during execution'):validate_source_snapshot(root,recorded)

    def test_changed_retained_training_data_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            _,scope,manifest=self.fixture(directory);(scope/'train.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Retained SWD changed'):runner.audit_engine(manifest)

    def test_cleanup_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run,scope,manifest=self.fixture(directory);outside=Path(directory)/'original.csv';outside.write_text('original')
            write_json(scope/'projection_cleanup.json',{'status':'HASHED_FOR_REMOVAL','files':[{'path':str(outside),'sha256':sha256(outside)}]})
            with self.assertRaises(ValueError):runner.cleanup_projection(scope,[manifest],run)
            self.assertEqual(outside.read_text(),'original')

    def test_partial_attempt_is_preserved_and_next_attempt_is_new(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'scope';first=runner.attempt(base);first.mkdir();(first/'failure.json').write_text('evidence')
            second=runner.attempt(base)
            self.assertNotEqual(first,second);self.assertFalse(second.exists())
            self.assertEqual((first/'failure.json').read_text(),'evidence')

    def test_predicted_coordinate_order_and_cloglog_scale_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'projection.csv'
            pd.DataFrame({'longitude':[114.,114.1],'latitude':[34.,34.1],'cloglog':[.2,.4]}).to_csv(path,index=False)
            engine={'prediction_files':[str(path)],'prediction_rows':2}
            np.testing.assert_array_equal(prediction_values(engine,np.array([[114.,34.],[114.1,34.1]])),[.2,.4])
            with self.assertRaisesRegex(ValueError,'order/coordinates'):prediction_values(engine,np.array([[114.1,34.1],[114.,34.]]))
            pd.DataFrame({'longitude':[114.],'latitude':[34.],'cloglog':[1.1]}).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Invalid cloglog'):prediction_values(engine)

    def test_runner_rejects_policy_allowing_locked_test_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'inputs').mkdir();write_json(root/'inputs/manifest.json',{})
            write_json(root/'config.json',{'locked_test':{'evaluation':'evaluate_again'}})
            args=SimpleNamespace(run=root/'local_work/run',config=root/'config.json',inputs=root/'inputs',splits=root/'split')
            with patch.object(runner,'__file__',str(root/'scripts/zhengzhou/runner.py')),patch.object(runner.argparse.ArgumentParser,'parse_args',return_value=args),\
                    patch.object(runner,'verify_split_inventory',return_value={}),patch.object(runner,'run_jobs',side_effect=AssertionError('No fit permitted')):
                with self.assertRaisesRegex(ValueError,'Test isolation policy required'):runner.main()

    def test_runner_rejects_changed_resume_identity_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'inputs').mkdir();(root/'local_work/run/manifests').mkdir(parents=True)
            write_json(root/'inputs/manifest.json',{'locked_test_responses_in_output_tables':False,'config_sha256':'hash','outputs':{}})
            write_json(root/'config.json',{'locked_test':{'evaluation':'DISABLED_PREVIOUS_TEST_ALREADY_CONSUMED'},'complete_pool':{'parent_split_manifest_sha256':'hash'}})
            write_json(root/'local_work/run/manifests/run_manifest.json',{'config_sha256':'different'})
            table=pd.DataFrame({'raster_row':pd.Series(dtype=int),'raster_col':pd.Series(dtype=int)})
            ctx={'names':[str(i) for i in range(25)],**{k:table for k in ('presence','B0','reference','visits')}}
            args=SimpleNamespace(run=root/'local_work/run',config=root/'config.json',inputs=root/'inputs',splits=root/'split',resume=True)
            with patch.object(runner,'__file__',str(root/'scripts/zhengzhou/runner.py')),patch.object(runner.argparse.ArgumentParser,'parse_args',return_value=args),\
                    patch.object(runner,'verify_split_inventory',return_value={}),patch.object(runner,'development_split_plan',return_value={'development_fit_mask':np.ones((1,1),bool)}),\
                    patch.object(runner,'sha256',return_value='hash'),patch.object(runner,'context',return_value=ctx),\
                    patch.object(runner,'run_jobs',side_effect=AssertionError('No fit permitted')):
                with self.assertRaisesRegex(ValueError,'Resume identity differs'):runner.main()


if __name__=='__main__':unittest.main()
