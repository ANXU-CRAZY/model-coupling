"""Complete supplied-pool nested MaxEnt, development only, with resumable scopes.

No locked-response access, locked predictions, final-test claim or test metrics.
The parent partitions are retained; upstream predictor curation is unverified.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import pandas as pd

from scripts.zhengzhou.run_maxent_nested_cv import (
    SEASONS, VARIANTS, SCHEMES, stamp, seeded, context, prepare_fit, job,
    result_metrics, final_feature_plan)
from scripts.zhengzhou.export_maxent_oof import verify_split_inventory
from wetland_coupling.maxent_engine import check_jar, run_jobs, validate_scale
from wetland_coupling.maxent_protocol import (
    sha256, write_json, aggregate_candidates, choose_candidate, assert_oof_groups)
from wetland_coupling.maxent_splits import validate_split_plan


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def development_split_plan(splits):
    """Read partition metadata and masks, without deserializing bird CSVs."""
    verify_split_inventory(splits)
    plan=load(splits/'split_plan.json')
    with np.load(splits/'split_masks.npz',allow_pickle=False) as archive:
        plan['masks']={key:archive[key] for key in archive.files}
    validate_split_plan(plan)
    for outer in plan['outer_folds']:
        for fold in [outer,*outer['inner_folds']]:
            fold['fit_mask']=plan['masks'][fold['fit_mask_key']]
            fold['validation_mask']=plan['masks'][fold['validation_mask_key']]
    for fold in plan['final_inner_folds']:
        fold['fit_mask']=plan['masks'][fold['fit_mask_key']]
        fold['validation_mask']=plan['masks'][fold['validation_mask_key']]
    plan['development_fit_mask']=plan['masks']['development_fit']
    plan['locked_mask']=plan['masks']['locked_test']
    plan['development_groups']=sorted(np.unique(plan['masks']['group_raster'][plan['development_fit_mask']]).tolist())
    return plan


def audit_engine(manifest):
    """Retained output hashes or explicit reconstruction receipts are required."""
    path=Path(manifest);data=load(path)
    if data['status']!='OFFICIAL_MAXENT_FITTED':raise ValueError('Incomplete model: '+str(path))
    receipt=path.parent.parent/'projection_cleanup.json'
    removed={r['path']:r for r in load(receipt)['files']} if receipt.exists() else {}
    for name,meta in data['outputs'].items():
        item=path.parent/name
        if item.exists():
            if sha256(item)!=meta['sha256']:raise ValueError('Model output changed: '+str(item))
        elif str(item.resolve()) not in removed or removed[str(item.resolve())]['sha256']!=meta['sha256']:
            raise ValueError('Unaccounted missing output: '+str(item))
    for key in ('train','background'):
        item=Path(data['input_paths'][key+'_csv'])
        if sha256(item)!=data['inputs'][key]['sha256']:raise ValueError('Retained SWD changed')
    return data


def cleanup_projection(scope, manifests, run):
    """Delete only reconstructed projection SWDs, after successful output audits."""
    receipt=scope/'projection_cleanup.json'
    if receipt.exists():
        for item in load(receipt)['files']:
            path=Path(item['path']);path.resolve().relative_to(run.resolve())
            if path.exists():
                if sha256(path)!=item['sha256']:raise ValueError('Pending cleanup source changed')
                path.unlink()
        data=load(receipt);data['status']='REMOVED';write_json(receipt,data)
        return
    for manifest in manifests:audit_engine(manifest)
    candidates=[scope/'projection.csv']
    for manifest in manifests:candidates.extend(sorted((Path(manifest).parent/'projection_inputs').glob('*.csv')))
    files=[]
    for path in candidates:
        path=path.resolve();path.relative_to(run.resolve())
        if path.is_file():files.append({'path':str(path),'sha256':sha256(path),'bytes':path.stat().st_size})
    data={'status':'HASHED_FOR_REMOVAL','created_at_utc':stamp(),'files':files,
          'reason':'Projection inputs reconstructable from hashed frozen inputs, selection, masks and offsets; official prediction CSVs retained',
          'bytes':sum(i['bytes'] for i in files),'model_manifests':[str(p) for p in manifests]}
    write_json(receipt,data)
    for item in files:Path(item['path']).unlink()
    data.update(status='REMOVED',ended_at_utc=stamp());write_json(receipt,data)


def attempt(base):
    base.mkdir(parents=True,exist_ok=True)
    index=1
    while (base/f'attempt_{index:03d}').exists():index+=1
    return base/f'attempt_{index:03d}'


def tuning(run,season,label,ctx,folds,config,args,progress):
    rows=[]
    for inner,fold in enumerate(folds):
        for background in SCHEMES:
            for variant in VARIANTS:
                base=run/variant/season/label/f'inner_{inner}'/background
                checkpoint=base/'completed.json'
                if checkpoint.exists():
                    done=load(checkpoint)
                    for row in done['rows']:audit_engine(row['artifact'])
                    cleanup_projection(Path(done['scope']),[r['artifact'] for r in done['rows']],run)
                    rows.extend(done['rows']);continue
                seed=seeded(config['seed'],f'{season}/{label}/{inner}/{background}')
                scope=attempt(base)
                progress(phase='inner_tuning',season=season,scope=label,inner=inner,background=background,variant=variant)
                meta=prepare_fit(scope,ctx,fold['fit_mask'],fold['validation_mask'],background,variant,config,seed)
                jobs=[job(scope,rm,fc,seed,args,config,f'rm{rm:g}_{fc}') for rm in config['maxent']['rm'] for fc in config['maxent']['fc']]
                results=run_jobs(jobs,max_workers=1)
                complete=[]
                for task,result in zip(jobs,results):
                    complete.append({'season':season,'scope':label,'inner_fold':inner,'variant':variant,
                        'background':background,'rm':task['rm'],'fc':task['fc'],**result_metrics(result,meta),
                        'predictors':';'.join(meta['predictors']),'candidate_predictor_count':23 if variant=='no_lst' else 25,
                        'fit_groups':fold['fit_groups'],'validation_groups':fold['validation_groups'],'artifact':result['manifest_path']})
                write_json(checkpoint,{'scope':str(scope.resolve()),'rows':complete,'completed_at_utc':stamp()})
                cleanup_projection(scope,[r['artifact'] for r in complete],run)
                rows.extend(complete)
                pd.DataFrame(rows).to_csv(run/'reports'/f'inner_{season}_{label}.csv',index=False)
                progress(completed_models_delta=len(complete))
                del results;gc.collect()
    pd.DataFrame(rows).to_csv(run/'reports'/f'inner_{season}_{label}.csv',index=False)
    aggregate=aggregate_candidates(rows)
    limit=config['selection']['mean_omission_eligibility_max']
    winners={v:{b:choose_candidate([r for r in aggregate if r['variant']==v and r['background']==b],limit) for b in SCHEMES} for v in VARIANTS}
    chosen={v:choose_candidate([r for r in aggregate if r['variant']==v],limit) for v in VARIANTS}
    write_json(run/'reports'/f'tuning_{season}_{label}.json',{'all_candidates':aggregate,'scheme_winners':winners,'variant_winners':chosen})
    return winners,chosen,aggregate


def member(run,season,label,ctx,fold,variant,candidate,config,args,progress):
    background=candidate['background'];base=run/variant/season/label/'refit'/background
    checkpoint=base/'completed.json'
    if checkpoint.exists():
        done=load(checkpoint);engine=audit_engine(done['provenance']['engine_manifest'])
        if sha256(done['provenance']['engine_manifest'])!=done['provenance']['engine_manifest_sha256']:raise ValueError('Refit provenance changed')
        values=np.concatenate([validate_scale(pd.read_csv(p).iloc[:,-1].to_numpy(float)) for p in engine['prediction_files']])
        cleanup_projection(Path(done['scope']),[done['provenance']['engine_manifest']],run)
    else:
        seed=seeded(config['seed'],f'{season}/{label}/refit/{background}');scope=attempt(base)
        progress(phase='outer_refit',season=season,scope=label,variant=variant,background=background)
        meta=prepare_fit(scope,ctx,fold['fit_mask'],fold['validation_mask'],background,variant,config,seed,True)
        result=run_jobs([job(scope,candidate['rm'],candidate['fc'],seed,args,config,'official_model')],max_workers=1)[0]
        provenance={'season':season,'scope':label,'variant':variant,'background':background,'rm':candidate['rm'],'fc':candidate['fc'],
            'predictors':meta['predictors'],'candidate_predictor_count':23 if variant=='no_lst' else 25,
            'fit_groups':fold['fit_groups'],'tune_groups':fold['fit_groups'],'validation_groups':fold['validation_groups'],
            'calibrate_groups':[],'engine_manifest':result['manifest_path'],'engine_manifest_sha256':sha256(result['manifest_path']),
            'output_scale':'cloglog','background_is_absence':False,'candidate_provenance':config['predictors']['candidate_provenance'],
            'screening_uses_fit_background_only':True,'strict_end_to_end_oof':False,'gate_eligible':False}
        done={'scope':str(scope.resolve()),'meta':{k:v for k,v in meta.items() if k!='raster_indices'},
              'metrics':result_metrics(result,meta),'provenance':provenance,'completed_at_utc':stamp()}
        write_json(scope/'provenance.json',provenance);write_json(checkpoint,done)
        values=result['predictions'];cleanup_projection(scope,[result['manifest_path']],run)
        progress(completed_models_delta=1)
    indices=np.flatnonzero(fold['validation_mask']);meta=done['meta']
    if len(values)!=meta['projection_start']+len(indices):raise ValueError('Refit projection length changed')
    return values[meta['projection_start']:],indices,done


def save_raster(path,values,ctx):
    import rasterio
    from affine import Affine
    grid=ctx['grid'];arr=np.asarray(values,dtype=np.float32).reshape(ctx['env'].shape[:2])
    with rasterio.open(path,'w',driver='GTiff',height=grid['height'],width=grid['width'],count=1,dtype='float32',
        crs='EPSG:32649',transform=Affine(*grid['transform']),nodata=-9999,compress='deflate',tiled=True) as dst:
        dst.write(np.where(np.isfinite(arr),arr,-9999).astype(np.float32),1)
        dst.update_tags(output_scale='cloglog',calibrated_probability='false',scope='candidate_community',
            oof_condition='full_supplied_pool_screened_within_folds_upstream_curation_unverified',gate_eligible='false')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','run','inputs','splits'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--java',required=True);parser.add_argument('--jar',required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];run=args.run.resolve();run.relative_to(root/'local_work')
    config=load(args.config);inputs=load(args.inputs/'manifest.json');split=verify_split_inventory(args.splits)
    if config['locked_test']['evaluation']!='DISABLED_PREVIOUS_TEST_ALREADY_CONSUMED':raise ValueError('Test isolation policy required')
    if inputs['locked_test_responses_in_output_tables'] or inputs['config_sha256']!=sha256(args.config):raise ValueError('Unexpected input preparation')
    if sha256(args.splits/'manifest.json')!=config['complete_pool']['parent_split_manifest_sha256']:raise ValueError('Parent split changed')
    for name,item in inputs['outputs'].items():
        if sha256(args.inputs/name)!=item['sha256']:raise ValueError('Prepared input changed: '+name)
    plan=development_split_plan(args.splits)
    for season in SEASONS:
        ctx=context(args.inputs,season)
        if len(ctx['names'])!=25:raise ValueError('Incomplete pool')
        for key in ('presence','B0','reference','visits'):
            table=ctx[key]
            if not plan['development_fit_mask'][table.raster_row.to_numpy(int),table.raster_col.to_numpy(int)].all():raise ValueError('Nondevelopment table row')
        del ctx
    code=[Path(__file__).resolve(),root/'scripts/zhengzhou/run_maxent_nested_cv.py',root/'scripts/zhengzhou/MaxentBatch.java',*(root/'wetland_coupling').glob('maxent_*.py')]
    identity={'config_sha256':sha256(args.config),'input_manifest_sha256':sha256(args.inputs/'manifest.json'),
        'split_manifest_sha256':sha256(args.splits/'manifest.json'),'source_code_sha256':{str(p.relative_to(root)):sha256(p) for p in code}}
    manifest=run/'manifests/run_manifest.json';lock=run/'manifests/ACTIVE_PROCESS.json'
    manifest.parent.mkdir(parents=True,exist_ok=True)
    if manifest.exists():
        if not args.resume:raise FileExistsError('Existing run requires --resume')
        state=load(manifest)
        if any(state.get(k)!=v for k,v in identity.items()):raise ValueError('Resume identity differs')
        if state['status']=='DEVELOPMENT_NESTED_CV_COMPLETE':raise ValueError('Run already complete')
    else:
        if args.resume:raise ValueError('No run to resume')
        state={**identity,'started_at_utc':stamp(),'python':platform.python_version(),'argv':sys.argv,
            'runtime':check_jar(args.java,args.jar),'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip(),
            'canonical_split_sha256':split['split_hash'],'locked_test_used':False,'locked_response_tables_available':False,
            'strict_end_to_end_oof':False,'gate_eligible':False,'planned_models':3528,'completed_models':0,'seasons_complete':[]}
        write_json(run/'manifests/config_snapshot.json',config)
    if lock.exists():
        old=load(lock);pid=int(old['pid'])
        alive=subprocess.run(['powershell','-NoProfile','-Command',f'Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id'],capture_output=True,text=True)
        if alive.stdout.strip():raise RuntimeError('Existing run process is still alive')
        if not args.resume:raise RuntimeError('Stale lock requires explicit resume')
        lock.unlink()
    with lock.open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'started_at_utc':stamp()},stream)
    try:
        for folder in (*VARIANTS,'oof','reports'):(run/folder).mkdir(exist_ok=True)
        state.update(status='RUNNING',resumed_at_utc=stamp() if args.resume else None)
        state['completed_models']=sum(len(done['rows']) if 'rows' in done else 1 for done in
            (load(p) for p in run.rglob('completed.json')))
        def progress(**values):
            state['completed_models']+=values.pop('completed_models_delta',0)
            state['progress']={**state.get('progress',{}),**values,'at_utc':stamp()}
            write_json(manifest,state);print(json.dumps({'completed_models':state['completed_models'],**state['progress']}),flush=True)
        progress(phase='preflight_complete')
        rows=[];artifacts=[];selections={}
        for season in SEASONS:
            ctx=context(args.inputs,season);size=ctx['env'].shape[0]*ctx['env'].shape[1]
            oof={v:{'M_oof':np.full(size,np.nan,dtype=np.float32),'q05':np.full(size,np.nan,dtype=np.float32),
                'median':np.full(size,np.nan,dtype=np.float32),'q95':np.full(size,np.nan,dtype=np.float32),
                'std':np.full(size,np.nan,dtype=np.float32),'member_count':np.zeros(size,dtype=np.int16),'fold':np.full(size,-1,dtype=np.int16)} for v in VARIANTS}
            for fold in plan['outer_folds']:
                label=f"outer_{fold['fold']}"
                assert_oof_groups(fold['validation_groups'],fold['fit_groups'],fold['fit_groups'],plan['locked_groups'])
                winners,chosen,_=tuning(run,season,label,ctx,fold['inner_folds'],config,args,progress)
                for variant in VARIANTS:
                    members=[];ids=[]
                    for background in SCHEMES:
                        prediction,indices,done=member(run,season,label,ctx,fold,variant,winners[variant][background],config,args,progress)
                        members.append(prediction);ids.append(done['provenance']['engine_manifest']);artifacts.append(done['provenance'])
                        candidate=winners[variant][background]
                        rows.append({'season':season,'fold':fold['fold'],'variant':variant,'background':background,
                            'rm':candidate['rm'],'fc':candidate['fc'],'chosen_background':background==chosen[variant]['background'],
                            **done['metrics'],'predictors':';'.join(done['meta']['predictors']),'artifact':ids[-1]})
                    values=np.stack(members);out=oof[variant];primary=list(SCHEMES).index(chosen[variant]['background'])
                    out['M_oof'][indices]=values[primary]
                    for key,val in zip(('q05','median','q95'),np.quantile(values,[.05,.5,.95],axis=0)):out[key][indices]=val
                    out['std'][indices]=values.std(axis=0,ddof=0);out['member_count'][indices]=3;out['fold'][indices]=fold['fold']
                    write_json(run/'oof'/f'{season}_{variant}_{label}_members.json',{'members':ids,'primary':ids[primary],
                        'validation_groups':fold['validation_groups'],'fit_and_tune_groups':fold['fit_groups'],
                        'split_sha256':identity['split_manifest_sha256'],'strict_end_to_end_oof':False,'gate_eligible':False})
                pd.DataFrame(rows).to_csv(run/'reports/outer_metrics.csv',index=False)
            for variant,out in oof.items():
                if not np.array_equal(np.isfinite(out['M_oof']),plan['development_fit_mask'].ravel()):raise ValueError('Unexpected OOF coverage')
                if np.isfinite(out['M_oof'][plan['locked_mask'].ravel()]).any():raise ValueError('Locked prediction produced')
                np.savez_compressed(run/'oof'/f'{season}_{variant}.npz',**out)
                save_raster(run/'oof'/f'{season}_{variant}_M_oof.tif',out['M_oof'],ctx)
                save_raster(run/'oof'/f'{season}_{variant}_U_M.tif',out['std'],ctx)
            del oof;gc.collect()
            _,chosen,aggregate=tuning(run,season,'full_development',ctx,plan['final_inner_folds'],config,args,progress)
            features={v:final_feature_plan(ctx,plan['development_fit_mask'],c['background'],v,config,
                seeded(config['seed'],f"{season}/frozen_final/{c['background']}")) for v,c in chosen.items()}
            selections[season]={'variant_winners':chosen,'deployment_choice':choose_candidate(aggregate,config['selection']['mean_omission_eligibility_max']),'final_features':features}
            if season not in state['seasons_complete']:state['seasons_complete'].append(season)
            progress(phase='season_complete',season=season);del ctx;gc.collect()
        verify_split_inventory(args.splits)
        if sha256(args.inputs/'manifest.json')!=identity['input_manifest_sha256']:raise ValueError('Input manifest changed')
        write_json(run/'manifests/development_selection.json',{'selections':selections,'selection_uses_only_inner_validation':True,'locked_test_used':False,**identity})
        write_json(run/'manifests/model_provenance.json',{'artifacts':artifacts,**identity})
        if state['completed_models']!=3528:raise ValueError('Unexpected completed checkpoint count')
        state.update(status='DEVELOPMENT_NESTED_CV_COMPLETE',ended_at_utc=stamp(),
            oof_ready='CONDITIONAL_ON_COMPLETE_SUPPLIED_POOL_UPSTREAM_PROVENANCE_UNVERIFIED',confidence_intervals=False)
        state['outputs']={str(p.relative_to(run)):{'sha256':sha256(p),'bytes':p.stat().st_size} for folder in ('oof','reports') for p in (run/folder).rglob('*') if p.is_file()}
        write_json(manifest,state);print(json.dumps({'status':state['status'],'locked_test_used':False}),flush=True)
    except BaseException as error:
        state.update(status='FAILED_OR_INCOMPLETE',failed_at_utc=stamp(),failure=repr(error));write_json(manifest,state)
        raise
    finally:
        if lock.exists():lock.unlink()


if __name__=='__main__':main()
